# app/travel_system/utils/pdf_saver.py

import os
import re
import logging 
import requests 
import io
import tempfile    
from pathlib import Path 
from fpdf import FPDF

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')

class TravelPDFSaver:
    def __init__(self, font_dir='assets/fonts', output_dir='plans'):
        # Get absolute paths of base directories with Path object
        base_dir = Path(font_dir).resolve() 
        output_base_dir = Path(output_dir).resolve()

        self.font_dir = str(base_dir) 
        self.output_dir = str(output_base_dir)

        # Store absolute Path objects for font files
        self.regular_font_filename = 'DejaVuSansCondensed.ttf'
        self.bold_font_filename = 'DejaVuSansCondensed-Bold.ttf'
        self.regular_font_path_obj = base_dir / self.regular_font_filename
        self.bold_font_path_obj = base_dir / self.bold_font_filename
        
        # Check font file existence during initialization
        if not self.regular_font_path_obj.is_file():
             logging.error(f"Initialization: Regular font not found! Path: {self.regular_font_path_obj}")
             raise FileNotFoundError(f"Regular font not found at {self.regular_font_path_obj}. Please ensure fonts are in {self.font_dir}")
        if not self.bold_font_path_obj.is_file():
             logging.error(f"Initialization: Bold font not found! Path: {self.bold_font_path_obj}")
             raise FileNotFoundError(f"Bold font not found at {self.bold_font_path_obj}. Please ensure fonts are in {self.font_dir}")

        logging.info(f"PDF Saver initialized. Font directory (for verification): {self.font_dir}")
        logging.info(f"Regular font Path object to be used: {self.regular_font_path_obj}")
        logging.info(f"Bold font Path object to be used: {self.bold_font_path_obj}")

        # Basic settings for PDF creation
        self.default_font_size = 11
        self.line_height = 6 # mm

        # Create output directory
        output_base_dir.mkdir(parents=True, exist_ok=True)

    # Function to extract title from text
    def extract_title(self, text):
        match = re.search(r'\*\*(.*?)\*\*', text)
        if match:
            title = match.group(1).strip()
            safe_title = re.sub(r'[\\/*?:"<>|]', "", title)
            safe_title = safe_title.replace(" ", "_")
            safe_title = safe_title[:50]
            # return statement should stay inside if block
            if safe_title:
                 return safe_title
            else:
                 return "travel_plan" # If title cannot be extracted or is empty
        return "travel_plan" # If no match

    # Function to generate unique PDF filename
    def generate_unique_filename(self, base_name="Travel_plan"):
        if base_name.lower().endswith('.pdf'):
            base_name = base_name[:-4]

        output_path_obj = Path(self.output_dir)
        counter = 1
        while True:
            filename = f"{base_name}_{counter}.pdf"
            file_path = output_path_obj / filename
            if not file_path.exists():
                return filename
            counter += 1

    # Process map URL and save as temporary file
    def download_map_image(self, map_url):
        try:
            # Download image
            headers = {'User-Agent': 'Mozilla/5.0'} 
            response = requests.get(map_url, stream=True, timeout=15, headers=headers) 
            response.raise_for_status() 
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            image_type = 'PNG'; 
            if 'png' in content_type: image_type = 'PNG'
            elif 'jpeg' in content_type or 'jpg' in content_type: image_type = 'JPG'
            elif content_type: logging.warning(f"Unsupported map format: {content_type}. Will default to PNG.")
            else: logging.warning("Could not get map content type. Defaulting to PNG.")
            
            # Create temporary file and write content
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{image_type.lower()}')
            temp_file.write(response.content)
            temp_file.close()
            
            logging.info(f"Map image saved to temporary file: {temp_file.name}")
            return temp_file.name, image_type
        except Exception as e:
            logging.error(f"Error downloading map: {e}", exc_info=True)
            return None, None

    # Main PDF creation function
    def save_travel_plan_to_pdf(self, plan_text, filename=None):
        
        pdf = FPDF()
        pdf.add_page()

        try:
            # Get absolute path, fix slashes, and check existence ONE LAST TIME
            regular_font_path_str = str(self.regular_font_path_obj.resolve()).replace('\\', '/')
            bold_font_path_str = str(self.bold_font_path_obj.resolve()).replace('\\', '/')
            
            logging.info(f"Regular Path for add_font: {regular_font_path_str}")
            if not os.path.exists(regular_font_path_str):
                 logging.error(f"ERROR: Regular font NOT FOUND before add_font: {regular_font_path_str}")
                 raise FileNotFoundError(f"Cannot find regular font right before add_font: {regular_font_path_str}")
            pdf.add_font('DejaVu', '', regular_font_path_str)

            logging.info(f"Bold Path for add_font: {bold_font_path_str}")
            if not os.path.exists(bold_font_path_str):
                 logging.error(f"ERROR: Bold font NOT FOUND before add_font: {bold_font_path_str}")
                 raise FileNotFoundError(f"Cannot find bold font right before add_font: {bold_font_path_str}")
            pdf.add_font('DejaVu', 'B', bold_font_path_str)

            # Using normal font for italics
            pdf.add_font('DejaVu', 'I', regular_font_path_str)
            
            logging.info("fpdf.add_font calls made successfully (with absolute path).")
        except Exception as font_err:
             logging.error(f"CRITICAL error occurred while adding fpdf fonts: {font_err}", exc_info=True)
             raise RuntimeError(f"PDF fonts could not be added (despite absolute path attempt): {font_err}") from font_err

        # First find and download all map URLs
        map_url_pattern = r"(https?://api\.tomtom\.com/map/1/staticimage[^\s]+)"
        map_matches = re.findall(map_url_pattern, plan_text)
        map_files = []
        
        # Download all map URLs and save to temporary files
        for map_url in map_matches:
            temp_file_path, image_type = self.download_map_image(map_url)
            if temp_file_path:
                map_files.append((map_url, temp_file_path, image_type))
                logging.info(f"Temporary file prepared for map URL {map_url}.")
            else:
                logging.error(f"Could not download map URL {map_url}.")
                
        # Now process text line by line
        lines = plan_text.strip().split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                pdf.ln(self.line_height / 2)
                continue
            
            # Detect line containing map reference
            map_url = None
            for url, _, _ in map_files:
                if url in line:
                    map_url = url
                    break
                    
            if map_url:
                # This line contains a map reference
                
                # Add text content excluding URL
                clean_text = re.sub(map_url_pattern, "", line).strip()
                if clean_text:
                    # Add clean text content
                    pdf.set_font('DejaVu', '', self.default_font_size)
                    if clean_text.endswith(":"):
                        pdf.multi_cell(0, self.line_height, clean_text, border=0, align='L')
                    else:
                        pdf.multi_cell(0, self.line_height, f"{clean_text}:", border=0, align='L')
                else:
                    # If no text, just write "Map view:"
                    pdf.set_font('DejaVu', '', self.default_font_size)
                    pdf.multi_cell(0, self.line_height, "Map view:", border=0, align='L')
                
                # Add map image
                map_index = next((i for i, (url, _, _) in enumerate(map_files) if url == map_url), None)
                if map_index is not None:
                    _, temp_file_path, image_type = map_files[map_index]
                    available_width = pdf.w - pdf.l_margin - pdf.r_margin
                    
                    try:
                        # Determine new position
                        pdf.ln(self.line_height * 0.5)
                        
                        # Add image
                        pdf.image(name=temp_file_path, type=image_type, w=available_width)
                        pdf.ln(self.line_height)
                        logging.info(f"Map image added to PDF: {temp_file_path}")
                        
                        # URL reference won't be added anymore
                    except Exception as img_err:
                        logging.error(f"Error adding map to PDF: {img_err}", exc_info=True)
                        pdf.multi_cell(0, self.line_height, f"[Map image could not be added]", border=0, align='L')
                continue  # Line processed, continue
            
            # Check for heading
            heading_match = re.match(r'^(\d+\.)\s*(\*\*.*?\*\*)', line)
            if heading_match:
                number_part = heading_match.group(1)
                bold_part = heading_match.group(2)
                text_inside = bold_part[2:-2].strip()
                pdf.set_font('DejaVu', 'B', self.default_font_size + 1) 
                pdf.multi_cell(0, self.line_height, f"{number_part} {text_inside}", border=0, align='L')
                pdf.ln(self.line_height * 0.5)
                continue # Go to next line

            # Check for bullet point
            elif line.startswith('* '):
                pdf.set_x(pdf.l_margin + 5) 
                pdf.set_font('DejaVu', '', self.default_font_size)
                content = line[2:]
                parts = re.split(r'(\*\*.*?\*\*)', content)
                is_first_part = True # Flag for space after Write
                for part in parts:
                    if not part: 
                        continue
                    # Add space after first part and if not at the beginning of line
                    if not is_first_part and pdf.get_x() > (pdf.l_margin + 5): 
                         pdf.write(self.line_height, " ") 
                         
                    if part.startswith('**') and part.endswith('**'):
                        pdf.set_font('DejaVu', 'B', self.default_font_size)
                        pdf.write(self.line_height, part[2:-2])
                        pdf.set_font('DejaVu', '', self.default_font_size)
                    else:
                        pdf.write(self.line_height, part)
                    is_first_part = False
                pdf.ln(self.line_height)
                pdf.set_x(pdf.l_margin) # Reset x for next possible bullet
                continue # Go to next line

            # Check for standalone bold heading (usually subheading)
            elif line.startswith('**') and line.endswith('**') and len(line) > 4:
                  pdf.set_font('DejaVu', 'B', self.default_font_size)
                  text_inside = line[2:-2]
                  pdf.multi_cell(0, self.line_height, text_inside, border=0, align='L')
                  pdf.ln(self.line_height * 0.3) # Less space after heading
                  continue # Go to next line
                  
            # Plain text or text with bold sections
            else: 
                  pdf.set_font('DejaVu', '', self.default_font_size)
                  parts = re.split(r'(\*\*.*?\*\*)', line)
                  is_first_part = True # Flag for space after Write
                  for part in parts:
                     if not part: 
                         continue
                     # Add space after first part and if not at the beginning of line
                     if not is_first_part and pdf.get_x() > pdf.l_margin : 
                          pdf.write(self.line_height, " ") 
                     if part.startswith('**') and part.endswith('**'):
                          pdf.set_font('DejaVu', 'B', self.default_font_size)
                          pdf.write(self.line_height, part[2:-2])
                          pdf.set_font('DejaVu', '', self.default_font_size)
                     else:
                          pdf.write(self.line_height, part)
                     is_first_part = False
                  pdf.ln(self.line_height) # End of line
                  continue # Go to next line

        if filename: 
            base_name = filename
        else: 
            base_name = self.extract_title(plan_text) # Use title
            
        final_filename = self.generate_unique_filename(base_name)
        output_path = Path(self.output_dir) / final_filename
        output_path_str = str(output_path)

        try:
            logging.info(f"Saving PDF as '{output_path_str}'...")
            pdf.output(output_path_str)
            print(f"Travel plan saved as '{output_path_str}'.") 
            
            # Clean up temporary files
            for _, temp_file_path, _ in map_files:
                try:
                    os.unlink(temp_file_path)
                    logging.info(f"Temporary file deleted: {temp_file_path}")
                except:
                    logging.warning(f"Could not delete temporary file: {temp_file_path}")
                    
            return output_path_str 
        except Exception as e:
            print(f"An error occurred while saving PDF file: {e}") 
            logging.error(f"Final error occurred while saving PDF file ({output_path_str}): {e}", exc_info=True)
            
            # Clean up temporary files even in case of error
            for _, temp_file_path, _ in map_files:
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                    
            raise RuntimeError(f"PDF could not be saved: {e}") from e