import zipfile
import xml.etree.ElementTree as ET
import os

def read_docx(file_path):
    try:
        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read('word/document.xml')
        
        tree = ET.fromstring(xml_content)
        
        # XML namespace for Word
        namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        
        text = []
        for node in tree.iter():
            if node.tag.endswith('}t'): # Text node
                if node.text:
                    text.append(node.text)
            elif node.tag.endswith('}p'): # Paragraph node
                text.append('\n')
        
        return ''.join(text)
    except Exception as e:
        return f"Error reading docx: {e}"

file_path = 'version 00.docx'
if os.path.exists(file_path):
    content = read_docx(file_path)
    print(content[:2000]) # Print first 2000 chars
else:
    print(f"File {file_path} not found.")
