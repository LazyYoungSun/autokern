import os
import tempfile
from PIL import Image, ImageDraw, ImageFont
from glyphsLib import GSFont
import glyphsLib
import ufo2ft

def glyphs_to_ttf(glyphs_path: str, target_master_id: str = None) -> str:
    """
    Конвертирует .glyphs в UFO, находит нужный мастер по ID 
    и компилирует в TTF только его, игнорируя тяжелый ExtraBlack и бэкапы.
    """
    print(f"Компиляция выбранного мастера из {os.path.basename(glyphs_path)} в TTF...")
    font = GSFont(glyphs_path)
    
    ufos = glyphsLib.to_ufos(font)
    
    selected_ufo = None
    if target_master_id:
        for ufo in ufos:
            ufo_master_id = ufo.lib.get("com.github.googlei18n.glyphsLib.masterId")
            if ufo_master_id == target_master_id:
                selected_ufo = ufo
                break

    if selected_ufo is None:
        selected_ufo = ufos[0]

    tmp_dir = tempfile.gettempdir()
    ttf_path = os.path.join(tmp_dir, "temp_autokern_font.ttf")
    
    compiled_ttf = ufo2ft.compileTTF(selected_ufo)
    compiled_ttf.save(ttf_path)
    
    return ttf_path

def rendering_kerning_pair(font_path, pair, font_size=150, img_size=224, crop_to_edge=True):
    font = ImageFont.truetype(font_path, font_size)
    
    if isinstance(pair, (tuple, list)):
        pair_str = "".join(pair)
        char1, char2 = pair[0], pair[1]
    else:
        pair_str = pair
        char1, char2 = pair[0], pair[1] if len(pair) > 1 else (pair[0], "")
        
    start_x, start_y = 20, 20
    
    img = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(img)
    draw.text((start_x, start_y), pair_str, font=font, fill=0)
    
    if not crop_to_edge:
        return img

    try:
        bbox1 = font.getmask(char1).getbbox()
        bbox2 = font.getmask(char2).getbbox()
        
        if not bbox1 or not bbox2:
            return img

        w1 = bbox1[2] - bbox1[0]
        w2 = bbox2[2] - bbox2[0]
        
        cut_left = start_x + bbox1[0] + (w1 // 2)
        adv_width1 = font.getlength(char1)
        cut_right = start_x + adv_width1 + bbox2[0] + (w2 // 2)
        
        cropped = img.crop((cut_left, 0, cut_right, img_size))
        
        final_img = Image.new("L", (img_size, img_size), 255)
        paste_x = (img_size - cropped.width) // 2
        final_img.paste(cropped, (paste_x, 0))
        
        return final_img
        
    except Exception as e:
        print(f"Ошибка обрезки для {pair_str}: {e}")
        return img
