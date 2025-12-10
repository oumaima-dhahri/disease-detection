# Guide: Exporting ConvNeXt MSF Architecture to Microsoft Word

## Option 1: Export as High-Resolution Image (Recommended)

### Method A: Using PlantUML Online Server
1. Go to http://www.plantuml.com/plantuml/uml/
2. Copy the contents of `convnext_msf_architecture.puml`
3. Paste into the online editor
4. Click "Submit" to generate the diagram
5. Right-click the image → "Save image as..." → Save as PNG
6. In Word: Insert → Pictures → Select the PNG file
7. Right-click image → "Format Picture" → Adjust size and ensure high quality

### Method B: Using VS Code Extension
1. Install "PlantUML" extension in VS Code
2. Open `convnext_msf_architecture.puml`
3. Press `Alt+D` (or right-click → "Preview PlantUML")
4. Right-click the preview → "Export Current Diagram" → Choose PNG/SVG
5. Insert into Word as above

### Method C: Using Command Line (Java required)
```bash
java -jar plantuml.jar -tpng -r convnext_msf_architecture.puml
```
This generates a PNG file you can insert into Word.

## Option 2: Export as SVG (Vector, Scalable)
1. Use any method above but export as SVG instead of PNG
2. In Word: Insert → Pictures → Select SVG
3. SVG scales without quality loss - perfect for printing

## Option 3: Copy-Paste from Rendered View
1. Render the diagram using PlantUML online or VS Code
2. Take a screenshot (Windows: Win+Shift+S)
3. Paste directly into Word
4. Right-click → "Format Picture" → Adjust quality

## Option 4: Create in Word Directly (Alternative)
If text clarity is an issue, you can:
1. Create the diagram structure in Word using:
   - Insert → Shapes (rectangles, arrows)
   - Insert → Text Boxes for labels
   - Use the same color scheme from the PlantUML script
2. Group all elements: Select all → Right-click → Group

## Improving Text Clarity in Word

After inserting the image:
1. **Right-click image → Format Picture**
2. **Picture Corrections:**
   - Increase Sharpness: +20 to +30
   - Adjust Brightness/Contrast if needed
3. **Compress Pictures:**
   - Uncheck "Apply only to this picture" if you want to compress all
   - Choose "High fidelity" for best quality
4. **Size:**
   - Lock aspect ratio
   - Set width to 6-7 inches for readability

## Recommended Settings for Word Document

1. **Page Setup:**
   - Orientation: Landscape (for horizontal layout)
   - Margins: Narrow (0.5 inch)

2. **Image Settings:**
   - Resolution: At least 300 DPI for printing
   - Format: PNG (lossless) or SVG (vector)
   - Size: Fit to page width (6-7 inches)

3. **Caption:**
   - Insert → Caption → "Figure X: ConvNeXt with Multi-Scale Fusion Architecture"

## Troubleshooting Text Clarity

If text is still unclear:
1. Export at higher resolution: Use `-tsvg` or increase DPI
2. Use larger font sizes in PlantUML (already set to 11-12pt)
3. Export as PDF first, then convert to image at high DPI
4. Consider splitting into multiple smaller diagrams

## Quick Export Script

Save this as `export_diagram.bat` (Windows):
```batch
@echo off
java -jar plantuml.jar -tpng -r -SDPI=300 convnext_msf_architecture.puml
echo Diagram exported as PNG at 300 DPI
pause
```

Or for SVG:
```batch
@echo off
java -jar plantuml.jar -tsvg -r convnext_msf_architecture.puml
echo Diagram exported as SVG
pause
```

