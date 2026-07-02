import os
import json
import click
import torch
from fontTools.ttLib import TTFont
from fontTools.feaLib.builder import addOpenTypeFeaturesFromString
from autokern.model import predict_kerning

def get_char_to_glyph_map(font: TTFont) -> dict:
    """
    Строит мапу {символ: имя_глифа_в_шрифте} на основе таблицы cmap шрифта.
    """
    cmap = font.getBestCmap()
    char_to_glyph = {}
    if cmap:
        for codepoint, glyph_name in cmap.items():
            try:
                char_to_glyph[chr(codepoint)] = glyph_name
            except Exception:
                continue
    return char_to_glyph

def extract_existing_kerning(font: TTFont) -> dict:
    """
    Программно извлекает весь существующий кернинг из таблиц GPOS шрифта.
    Возвращает словарь {('имя_глифа1', 'имя_глифа2'): значение}
    """
    existing_kerning = {}
    if 'GPOS' not in font:
        return existing_kerning
        
    gpos = font['GPOS'].table
    if not gpos or not gpos.LookupList:
        return existing_kerning
        
    for lookup in gpos.LookupList.Lookup:
        if lookup.LookupType == 2:
            for subtable in lookup.SubTable:
                if subtable.Format == 1:
                    for i, glyph_name1 in enumerate(subtable.Coverage.glyphs):
                        pair_set = subtable.PairSet[i]
                        for pair_value_record in pair_set.PairValueRecord:
                            glyph_name2 = pair_value_record.SecondGlyph
                            val = pair_value_record.Value1.XAdvance if pair_value_record.Value1 else 0
                            if val != 0:
                                existing_kerning[(glyph_name1, glyph_name2)] = int(val)
    return existing_kerning

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), help='Путь для сохранения нового .ttf файла')
@click.option('-j', '--json', 'json_path', type=click.Path(exists=True), help='Путь к кастомному JSON-файлу со списком пар')
@click.option('-p', '--pairs', type=str, help='Список пар строкой через запятую (например: -p "AV,VA,ьх")')
def main(input_file, output, json_path, pairs):
    """
    AutoKern CLI: Принимает .ttf шрифт, рассчитывает кернинг с помощью нейросети и выдает новый откерненный .ttf.
    """
    if not input_file.endswith('.ttf'):
        click.echo("Ошибка: Утилита принимает только файлы формата .ttf", err=True)
        return

    if not output:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_kerned{ext}"
    else:
        output_file = output

    pairs_list = []
    
    if pairs:
        click.echo(f"Парсинг пар из командной строки: {pairs}")
        raw_pairs = [p.strip() for p in pairs.split(",") if p.strip()]
        for rp in raw_pairs:
            if len(rp) != 2:
                click.echo(f"Ошибка: Комбинация '{rp}' некорректна. Пара должна состоять строго из 2 символов!", err=True)
                return
            pairs_list.append([rp[0], rp[1]])
    elif json_path:
        click.echo(f"Загрузка пользовательских пар из файла: {json_path}")
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                pairs_list = json.load(f)
        except Exception as e:
            click.echo(f"Ошибка чтения JSON: {e}", err=True)
            return
    else:
        current_dir = os.path.dirname(__file__)
        default_json_path = os.path.join(current_dir, 'good_latin_pairs_450.json')
        if not os.path.exists(default_json_path):
            click.echo(f"Ошибка: Встроенный –файл пар не найден по пути {default_json_path}", err=True)
            return
        with open(default_json_path, 'r', encoding='utf-8') as f:
            pairs_list = json.load(f)
        click.echo("Используется базовый набор из 450 латинских пар.")

    parsed_pairs = [tuple(p) for p in pairs_list]
    click.echo(f"Подготовлено пар для расчета: {len(parsed_pairs)}")

    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')

    font = TTFont(input_file)
    glyph_set = font.getGlyphSet()
    
    char_to_glyph = get_char_to_glyph_map(font)

    merged_kerning = extract_existing_kerning(font)
    click.echo(f"В исходном шрифте обнаружено {len(merged_kerning)} существующих пар кернинга.")

    click.echo("Ансамбль моделей рассчитывает кернинг...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        predictions = predict_kerning(parsed_pairs, input_file, weights_dir, device=device)
    except Exception as e:
        click.echo(f"Ошибка инференса: {e}", err=True)
        return

    new_pairs_added = 0
    overwritten_pairs = 0
    
    for (char1, char2), kern_val in predictions.items():
        g_name1 = char_to_glyph.get(char1, char1)
        g_name2 = char_to_glyph.get(char2, char2)
        
        if g_name1 in glyph_set and g_name2 in glyph_set:
            if (g_name1, g_name2) in merged_kerning:
                overwritten_pairs += 1
            else:
                new_pairs_added += 1
            merged_kerning[(g_name1, g_name2)] = kern_val

    click.echo(f"Результаты слияния: Добавлено новых пар: {new_pairs_added}, Перезаписано старых: {overwritten_pairs}")

    fea_text = "feature kern {\n"
    for (g_name1, g_name2), kern_val in merged_kerning.items():
        if g_name1 in glyph_set and g_name2 in glyph_set:
            fea_text += f"    pos {g_name1} {g_name2} {kern_val};\n"
    fea_text += "} kern;\n"

    try:
        addOpenTypeFeaturesFromString(font, fea_text)
        font.save(output_file)
        click.echo(click.style(f" Сгенерирован новый TTF: {output_file}", fg='green'))
    except Exception as e:
        click.echo(f"Ошибка при сборке OpenType таблиц: {e}", err=True)

if __name__ == '__main__':
    main()
