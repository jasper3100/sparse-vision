from PIL import Image, ImageDraw, ImageFont

# WE DO MANIPULATION ON PNG IMAGES USING PIL HERE!

'''
First, we shift the rightmost x-axis label of the rec loss vs sparsity plot to the left.
So that it does not extend beyond the plot boundaries. So that we can cut the plot sharply 
at this boundary to get rid of the superfluous second y-axis labelling.
'''

# Load the image
layer = "mixed3a"
epoch = [7,15,20,25,30,35,37,40,42,45,48,50] # excluded 3 --> 12 values --> [3x4] grid
# specify the plots where the x-axis tick labels only have two digits, because this changes the position of the x-axis label --> need to paint this area white
two_digit_x_axis_tick_labels = [15,25]  
grid_rows = 4
grid_columns = 3
# choose padding values that are divisible by 2
x_padding = 30
y_padding = 100
# choose space for y-axis label and x-axis label
x_label_padding = 200
y_label_padding = 220

#######################################################################
# DON'T CHANGE ANYTHING BELOW THIS LINE

# We select the desired rectangular area of the figure to crop.
crop_box = (3221, 119, 4598, 1442)
crop_height = crop_box[3] - crop_box[1]
crop_width = crop_box[2] - crop_box[0]
# add padding 
crop_height += x_padding # the padding for the x-axis label is along the y-axis
crop_width += y_padding

# Create a new blank image for the grid where we will paste the cropped images
grid_width = grid_columns * crop_width
grid_height = grid_rows * crop_height
# add space for y-axis label and x-axis label
grid_height += y_label_padding
grid_width += x_label_padding
grid_image = Image.new('RGB', (grid_width, grid_height), color="white")

font_file = r"C:\Users\Jasper\Downloads\Master thesis\code\dejavu-fonts-ttf-2.37\ttf\DejaVuSans.ttf"

def contains_black_pixels(image, rect):
    """
    Check if the rectangular area in the image contains black pixels.
    
    :param image: PIL Image object
    :param rect: Tuple (left, top, right, bottom) defining the rectangular area
    :return: True if the area contains black pixels, False otherwise
    """
    left, top, right, bottom = rect
    if left == right:
        right += 1 # otherwise we can't loop through anything
    for x in range(left, right):
        for y in range(top, bottom):
            pixel = image.getpixel((x, y))
            # if pixel is not white
            if pixel != (255, 255, 255, 255): # (R,G,B,A = alpha transparency), 
                return True
    return False

def shift_area(image, source_rect, shift_left):
    """
    Shift the specified rectangular area to the left by shift_left pixels.
    
    :param image: PIL Image object
    :param source_rect: Tuple (left, top, right, bottom) defining the rectangular area
    :param shift_left: Number of pixels to shift the area to the left
    :return: PIL Image object with the shifted area
    """
    left, top, right, bottom = source_rect
    # Extract the region to shift
    region = image.crop((left, top, right, bottom))
    # Create a new image for the result
    result = image.copy()
    # Paste the region back, shifted to the left
    result.paste(region, (left - shift_left, top))
    return result


counter = 0
for i in epoch:
    image_path = fr"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\{layer}_inceptionv1_1_0.001_512_sgd_sae_mlp_300_sae_eval_results_plot_nrmse_loss_epoch_{i}.png"

    image = Image.open(image_path)

    # Define the rectangular areas
    rect_to_check = (4599, 1330, 4599, 1440)  # (left, top, right, bottom)
    rect_to_shift = (4550, 1342, 4640, 1432)
    shift_left = 14  # Number of pixels to shift

    # Check for black pixels and shift the area if found
    if contains_black_pixels(image, rect_to_check):
        image = shift_area(image, rect_to_shift, shift_left)

    # Create a drawing context
    draw = ImageDraw.Draw(image) # Create a drawing context

    # draw white rectangle over the existing heading
    rect = (3380, 119, 4597, 186)
    draw.rectangle(rect, fill=(255, 255, 255))
    # then draw the new heading
    font_size = 54
    font = ImageFont.truetype(font_file, font_size) # DejaVuSans is the matplotlib default font
    # Add text to the image
    draw.text((3880, 124), f"Epoch {i}", font=font, fill="black")  # Black text # Alternatively within the plot: (4077, 225)
    
    if i in two_digit_x_axis_tick_labels:
        # draw white rectangle over the x-axis label if the x-axis tick labels only have two digits
        rect = (3460, 1420, 4511, 1480)
        draw.rectangle(rect, fill=(255, 255, 255))
        
    cropped_image = image.crop(crop_box)
    #cropped_image.show()

    x = (counter % grid_columns) * crop_width + int(x_padding / 2) + y_label_padding
    y = (counter // grid_columns) * crop_height + int(y_padding / 2)
    grid_image.paste(cropped_image, (x, y))

    grid_image_draw = ImageDraw.Draw(grid_image)
    font = ImageFont.truetype(font_file, 100)
    x_label = "1 - Sparsity"
    # include text at the center of the grid width --> compute size of text first
    _, _, w, h = grid_image_draw.textbbox((0, 0), x_label, font=font)
    grid_image_draw.text(((grid_width-w)/2, grid_height-130), x_label, font=font, fill="black")

    y_label = "Rec. loss (NRMSE)"
    _, _, w, h = grid_image_draw.textbbox((0, 0), y_label, font=font)
    # rotate text 90 degrees
    y_label_image = Image.new('RGB', (w, h), color="white")
    y_label_draw = ImageDraw.Draw(y_label_image)
    y_label_draw.text((0, 0), y_label, font=font, fill="black")
    y_label_image = y_label_image.rotate(90, expand=True)
    grid_image.paste(y_label_image, (30, int((grid_height-w)/2)-100))

    counter += 1

grid_image.show()

# Save the final grid image
output_path = fr"C:\Users\Jasper\Downloads\Master thesis\code\evaluation_results\inceptionv1\imagenet\{layer}_inceptionv1_1_0.001_512_sgd_sae_mlp_300_sae_eval_results_plot_nrmse_loss_rec_loss_vs_sparsity.png"
grid_image.save(output_path)
print(f'Grid image saved to {output_path}')