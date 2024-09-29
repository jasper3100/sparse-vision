from PIL import Image, ImageFont, ImageDraw

grid_columns = 3
grid_rows = 2
height_image = 399
width_image = 544
y_label_padding = 50
x_label_padding = 50

# Create a new blank image for the grid where we will paste the cropped images
grid_width = grid_columns * width_image
grid_height = grid_rows * height_image
# add space for y-axis label and x-axis label
grid_height += y_label_padding
grid_width += x_label_padding
grid_image = Image.new('RGB', (grid_width, grid_height), color="white")

font_file = r"C:\Users\Jasper\Downloads\Master thesis\code\dejavu-fonts-ttf-2.37\ttf\DejaVuSans.ttf"

epochs = [0, 10, 20, 30, 40, 50]
counter = 0
for epoch in epochs:
    file_path = fr"C:\Users\Jasper\Downloads\mixed3a_inceptionv1_1_0.001_512_sgd_gated_sae_0.001_256_constrained_adam_4_5.0_300_channel_frequency_histograms_epoch_{epoch}_cropped.png"
    image = Image.open(file_path)

    draw = ImageDraw.Draw(image) # Create a drawing context

    # add epoch label
    font_size = 20
    font = ImageFont.truetype(font_file, font_size) # DejaVuSans is the matplotlib default font
    # Add text to the image
    draw.text((245, 15), f"Epoch {epoch}", font=font, fill="black")  # Black text # Alternatively within the plot: (4077, 225)

    x = (counter % grid_columns) * width_image + y_label_padding
    y = (counter // grid_columns) * height_image
    grid_image.paste(image, (x, y))

    counter += 1

#'''
# Include x and y label
grid_image_draw = ImageDraw.Draw(grid_image)
font = ImageFont.truetype(font_file, 27)
x_label = "Activation frequency"
# include text at the center of the grid width --> compute size of text first
_, _, w, h = grid_image_draw.textbbox((0, 0), x_label, font=font)
grid_image_draw.text(((grid_width - x_label_padding -w)/2 + x_label_padding, grid_height-40), x_label, font=font, fill="black")

y_label = "No. of neurons"
_, _, w, h = grid_image_draw.textbbox((0, 0), y_label, font=font)
# rotate text 90 degrees
y_label_image = Image.new('RGB', (w, h), color="white")
y_label_draw = ImageDraw.Draw(y_label_image)
y_label_draw.text((0, 0), y_label, font=font, fill="black")
y_label_image = y_label_image.rotate(90, expand=True)
grid_image.paste(y_label_image, (15, int((grid_height - y_label_padding -w)/2)))
#'''

#grid_image.show()

# Save the final grid image
output_path = fr"C:\Users\Jasper\Downloads\mixed3a_inceptionv1_1_0.001_512_sgd_gated_sae_0.001_256_constrained_adam_4_5.0_300_channel_frequency_histograms_epochs_0_10_20_30_40_50_grid.png"
grid_image.save(output_path, quality=100)
print(f'Grid image saved to {output_path}')