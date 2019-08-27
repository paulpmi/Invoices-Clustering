import xml.etree.ElementTree as ET
import random
import cv2
import imgkit
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
from multiprocessing.dummy import Pool

number_of_images = 2000

BG_COLOR = 209
BG_SIGMA = 5
MONOCHROME = 1

text_corp = open('./processed_text.txt').read()

def blank_image(width=1024, height=1024, background=BG_COLOR):
    """
    It creates a blank image of the given background color
    """
    img = np.full((height, width, MONOCHROME), background, np.uint8)
    return img


def add_noise(img, sigma=BG_SIGMA):
    """
    Adds noise to the existing image
    """
    width, height, ch = img.shape
    n = noise(width, height, sigma=sigma)
    img = img + n
    return np.array(img.clip(0, 255), dtype=np.uint8)


def noise(width, height, ratio=1, sigma=BG_SIGMA):
    """
    The function generates an image, filled with gaussian nose. If ratio parameter is specified,
    noise will be generated for a lesser image and then it will be upscaled to the original size.
    In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
    uses interpolation.

    :param ratio: the size of generated noise "pixels"
    :param sigma: defines bounds of noise fluctuations
    """
    mean = 0
    assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(width, ratio)
    assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(height, ratio)

    h = int(height / ratio)
    w = int(width / ratio)

    result = np.random.normal(mean, sigma, (w, h, MONOCHROME))
    if ratio > 1:
        result = cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
    return result.reshape((width, height, MONOCHROME))


def texture(image, sigma=BG_SIGMA, turbulence=2):
    """
    Consequently applies noise patterns to the original image from big to small.

    sigma: defines bounds of noise fluctuations
    turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
    value - the more iterations will be performed during texture generation.
    """
    result = image.astype(float)
    cols, rows, ch = image.shape
    ratio = cols
    while not ratio == 1:
        result += noise(cols, rows, ratio, sigma=sigma)
        ratio = (ratio // turbulence) or 1
    cut = np.clip(result, 0, 255)
    return np.array(cut, dtype=np.uint8)


def generate_data(image_nr):

    coordinates = open('./table_images/table_{}.txt'.format(image_nr), 'w+')

    # Create a blank page and add some random text
    page = np.ones((1510, 1100), dtype=np.uint8) * 255
    font = ImageFont.truetype("./fonts/Helvetica.ttf", 20)
    im = Image.fromarray(page)
    draw = ImageDraw.Draw(im)
    for line in range(48):
        if random.random() < 0.2:
            message = " "
        else:
            message = text_corp[2000 * image_nr + 130 * line: 2000 * image_nr + 130 * (line + 1)]
        draw.text((50, 50 + 30 * line), message, font=font)

    page = np.array(im, dtype=np.uint8)
    page[:, 1050:] = 255
    tables = [(0, 0, 0, 0)]

    for table_nr in range(random.randint(1, 6)):

        body = ET.Element('div')

        # Randomly generate 4 table styles
        table_styles = ['classic', 'individual-cell', 'multiple-choice', 'multiple-choice-complex', 'header']
        style = random.choice(table_styles)

        if style == 'header':

            title_width = random.randint(120, 150)
            cells_first_row_left = random.randint(4, 8)
            cells_first_row_right = random.randint(4, 8)
            cells_second_row_left = random.randint(4, 8)
            width_second_row_right = random.randint(280, 350)
            cells_third_row = random.randint(10, 20)
            cells_forth_row = random.randint(10, 20)

            table1 = ET.SubElement(body, 'table')
            table1.set('style', 'font-size:16px; font-family: Arial, Helvetica, sans-serif; border-collapse: collapse')
            row = ET.SubElement(table1, 'tr')
            title = ET.SubElement(row, 'th')
            title.set('style', 'width: {}px'.format(title_width))
            title.text = 'Centre / Candidate No.'
            space = ET.SubElement(row, 'td')
            space.set('style', 'width: 5px')
            for c in range(cells_first_row_left):
                cell = ET.SubElement(row, 'td')
                cell.set('style', 'border: 1px solid black; width:30px; height: 40px')
            space = ET.SubElement(row, 'td')
            space.set('style', 'width: 15px')
            title = ET.SubElement(row, 'th')
            title.set('style', 'width: {}px'.format(title_width))
            title.text = 'Syllabus / Component'
            space = ET.SubElement(row, 'td')
            space.set('style', 'width: 5px')
            for c in range(cells_first_row_right):
                cell = ET.SubElement(row, 'td')
                cell.set('style', 'border: 1px solid black; width:30px; height: 40px')
            row = ET.SubElement(table1, 'tr')
            row.set('style', 'height:10px')

            table2 = ET.SubElement(body, 'table')
            table2.set('style', 'font-size:16px; font-family: Arial, Helvetica, sans-serif; border-collapse: collapse')
            row = ET.SubElement(table2, 'tr')
            title = ET.SubElement(row, 'th')
            title.set('style', 'width: {}px'.format(title_width))
            title.text = 'Version Number'
            space = ET.SubElement(row, 'td')
            space.set('style', 'width: 5px')
            for c in range(cells_second_row_left):
                cell = ET.SubElement(row, 'td')
                cell.set('style', 'border: 1px solid black; width:30px; height: 40px')
            space = ET.SubElement(row, 'td')
            space.set('style', 'width: 15px')
            title = ET.SubElement(row, 'th')
            title.set('style', 'width: {}px'.format(title_width))
            title.text = 'Examination title'
            space = ET.SubElement(row, 'td')
            space.set('style', 'width: 5px')
            cell = ET.SubElement(row, 'td')
            cell.set('style', 'border: 1px solid black; width: {}px; height: 40px'.format(width_second_row_right))
            row = ET.SubElement(table2, 'tr')
            row.set('style', 'height:10px')

            table3 = ET.SubElement(body, 'table')
            table3.set('style', 'font-size:16px; font-family: Arial, Helvetica, sans-serif; border-collapse: collapse')
            row = ET.SubElement(table3, 'tr')
            title = ET.SubElement(row, 'th')
            title.set('style', 'width: {}px'.format(title_width))
            title.text = 'First Name'
            space = ET.SubElement(row, 'td')
            space.set('style', 'width: 5px')
            for c in range(cells_third_row):
                cell = ET.SubElement(row, 'td')
                cell.set('style', 'border: 1px solid black; width:30px; height: 40px')
            row = ET.SubElement(table3, 'tr')
            row.set('style', 'height:10px')

            table4 = ET.SubElement(body, 'table')
            table4.set('style', 'font-size:16px; font-family: Arial, Helvetica, sans-serif; border-collapse: collapse')
            row = ET.SubElement(table4, 'tr')
            title = ET.SubElement(row, 'th')
            title.set('style', 'width: {}px'.format(title_width))
            title.text = 'Surname'
            space = ET.SubElement(row, 'td')
            space.set('style', 'width: 5px')
            for c in range(cells_forth_row):
                cell = ET.SubElement(row, 'td')
                cell.set('style', 'border: 1px solid black; width:30px; height: 40px')

            data = ET.tostring(body).decode("utf-8")


        if style == 'classic':

            table = ET.SubElement(body, 'table')

            columns_number = random.randint(4, 9)
            rows_number = random.randint(5, 10)
            row_height = random.randint(30, 60)
            table_width = random.randint(5, 10)*100
            border_width = random.randint(1, 3)
            space_between_rows = random.randint(0, 15)

            table.set('style','font-size:14px; border: {}px solid black; border-collapse: collapse; table-layout: fixed; width: {}px'.format(border_width,table_width))

            header = ET.SubElement(table, 'tr')
            for h in range(columns_number):
                column = ET.SubElement(header, 'th')
                column.set('style','padding-left: 10px; border: {}px solid black;  height: {}px'.format(border_width,row_height))
                position = random.randint(1, 1000)
                column.text = text_corp[position:position + int(table_width/10/columns_number)]
            row = ET.SubElement(table, 'tr')
            row.set('style', 'height: {}px'.format(space_between_rows))

            for r in range(rows_number - 1):
                row = ET.SubElement(table, 'tr')
                for c in range(columns_number):
                    column = ET.SubElement(row, 'td')
                    column.set('style','padding-left: 10px; border: {}px solid black; height: {}px'.format(border_width,row_height))
                    position = random.randint(1, 1000)
                    column.text = text_corp[position:position + int(table_width/10/columns_number)]

                row = ET.SubElement(table, 'tr')
                row.set('style', 'height: {}px'.format(space_between_rows))

            data = ET.tostring(table).decode("utf-8")


        if style == 'individual-cell':

            table = ET.SubElement(body, 'table')

            columns_number = random.randint(10, 20)
            rows_number = random.randint(5, 12)
            row_height = 35
            row_width = 30
            border_width = random.randint(1, 3)
            space_between_rows = random.randint(0, 30)
            space_between_columns = random.randint(0, 1)

            table.set('style', 'font-size:18px; border-collapse: collapse;'.format(space_between_columns))

            header = ET.SubElement(table, 'tr')
            for h in range(columns_number):
                column = ET.SubElement(header, 'th')
                column.set('style', 'border: {}px solid black; height: {}px; width: {}px; background: black'.format(border_width, row_height, row_width))
                if space_between_columns>0 and h<columns_number-1:
                    column = ET.SubElement(header, 'td')
                    column.set('style', 'width: {}px; border-top: {}px solid black; border-bottom: {}px solid black; background: black'.format(space_between_columns, border_width, border_width))
            row = ET.SubElement(table, 'tr')
            row.set('style', 'height: {}px'.format(space_between_rows))

            for r in range(rows_number - 1):
                row = ET.SubElement(table, 'tr')
                for c in range(columns_number):
                    if c == 0:
                        column = ET.SubElement(row, 'td')
                        column.set('style', 'font-weight: 600; text-align: center; border: {}px solid black; height: {}px; width: {}px'.format(border_width, row_height, row_width))
                        column.text = str(r + 1)
                    else:
                        column = ET.SubElement(row, 'td')
                        column.set('style', 'text-align: center; border: {}px solid black; height: {}px; width: {}px'.format(border_width, row_height, row_width))
                        column.text = random.choice('  ABCDEFGHIJKLMNOPQRSTUVWXYZ   ')
                    if space_between_columns>0 and c < columns_number - 1:
                        column = ET.SubElement(row, 'td')
                        column.set('style', 'width: {}px'.format(space_between_columns))

                row = ET.SubElement(table, 'tr')
                row.set('style', 'height: {}px'.format(space_between_rows))

            data = ET.tostring(table).decode("utf-8")


        if style == 'multiple-choice':

            table = ET.SubElement(body, 'table')

            columns_number = random.randint(4, 6)
            rows_number = random.randint(10, 15)
            row_height = 35
            row_width = 40
            border_width = random.randint(1, 3)
            space_between_rows = random.randint(0, 15)
            choice_tring = "ABCDE"

            table.set('style', 'font-size:18px; border-collapse: collapse;')

            for r in range(rows_number):
                row = ET.SubElement(table, 'tr')
                for c in range(columns_number):
                    if c == 0:
                        column = ET.SubElement(row, 'td')
                        column.set('style', 'font-weight: 600; text-align: center; border: {}px solid black; height: {}px; width: {}px'.format(border_width, row_height, row_width))
                        column.text = str(r + 1)
                    else:
                        column = ET.SubElement(row, 'td')
                        column.set('style', 'text-align: center; border: {}px solid black; height: {}px; width: {}px'.format(border_width, row_height, row_width))
                        column.text = choice_tring[c - 1]
                row = ET.SubElement(table, 'tr')
                row.set('style', 'height: {}px'.format(space_between_rows))

            data = ET.tostring(table).decode("utf-8")

        if style == 'multiple-choice-complex':

            table = ET.SubElement(body, 'table')

            columns_number = random.randint(4, 6)
            rows_number = random.randint(10, 15)
            row_height = 35
            row_width = 40
            border_width = random.randint(1, 3)
            space_between_rows = random.randint(0, 15)
            middle_region_width = random.randint(300, 500)
            choice_tring = "ABCDE"

            table.set('style', 'font-size:16px; font-family: Arial, Helvetica, sans-serif; border-collapse: collapse')

            for r in range(rows_number):
                row = ET.SubElement(table, 'tr')
                row.set('style', 'border: {}px solid black;'.format(border_width))
                for c in range(columns_number):
                    if c == 0:
                        column = ET.SubElement(row, 'td')
                        column.set('style', 'font-weight: 600; text-align: center; border: {}px solid black; height: {}px; width: {}px'.format(border_width, row_height, row_width))
                        column.text = str(r + 1)
                    else:
                        column = ET.SubElement(row, 'td')
                        column.set('style', 'text-align: center; border: {}px solid black; height: {}px; width: {}px'.format(0, row_height, row_width))
                        div = ET.SubElement(column, 'div')
                        span1 = ET.SubElement(div, 'span')
                        span1.text = choice_tring[c - 1]
                        ET.SubElement(div, 'br')
                        span2 = ET.SubElement(div, 'span')
                        span2.text = '&#9675;'

                column = ET.SubElement(row, 'td')
                column.set('style', 'border: {}px solid black; height: {}px; width: {}px'.format(border_width, row_height, middle_region_width))
                row = ET.SubElement(table, 'tr')
                row.set('style', 'height: {}px'.format(space_between_rows))

            data = ET.tostring(table).decode("utf-8")
            data = data.replace("&amp;", "&")


        # Convert the html table into an image and write coordinates for the table and each cell
        table_image = imgkit.from_string(data, False)
        table_image = np.fromstring(table_image, np.uint8)
        table_image = cv2.imdecode(table_image, cv2.IMREAD_COLOR)

        # Fix the imagekit margin issue - find table border and crop
        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
        x, y, w, h = cv2.boundingRect(thresh)
        table_image = gray[y:y + h, x:x + w]

        # Insert the table randomly into the page
        table_image = cv2.copyMakeBorder(table_image, 40, 40, 40, 40, cv2.BORDER_CONSTANT, None, (255, 255, 255))
        height, width = table_image.shape
        free_width = 1100 - width
        free_height = 1500 - height
        table_x = random.randint(20, free_width)
        table_y = random.randint(20, free_height)

        # Check if the newly inserted table intersects any of the other tables
        doesnt_intersect = True

        # determine the (x, y)-coordinates of the intersection rectangle
        boxA = (table_x, table_y, table_x + width, table_y + height)

        for boxB in tables:
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])

            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            if interArea > 0:
                doesnt_intersect = False

        if doesnt_intersect:
            page[table_y:table_y + height, table_x:table_x + width] = table_image

            # Save the coordinates for the current table (x, y, w, h) based on each style type
            # 40 and 80 values due to white border added to the tables
            if style == 'individual-cell':
                coordinates.write('table-{} {} {} {} {} \n'.format(table_nr, table_x + 40, table_y + 40 + row_height + space_between_rows, width - 80, height - 80 - row_height - space_between_rows))
                cell_width = int((width - 80 - 3 * space_between_columns * (columns_number - 1)) / (columns_number))
                cell_height = int((height - 80 - space_between_rows * (rows_number - 1)) / (rows_number))
                for row in range(1,rows_number):
                    for column in range(columns_number):
                        coordinates.write('cell-{}-{}-{} {} {} {} {} \n'.format(table_nr, row, column, column * (cell_width + 3*space_between_columns) + table_x + 40, row * (cell_height + space_between_rows) + table_y + 40, cell_width, cell_height))

            if style == 'header':
                coordinates.write('header-{} {} {} {} {} \n'.format(table_nr, table_x + 40, table_y + 40, width - 80, height - 80))

                row_height = int((height - 80 - 30)/4)
                first_row_left_width = title_width + 5 + 33 * cells_first_row_left
                first_row_right_width = title_width + 5 + 33 * cells_first_row_right
                second_row_left_width = title_width + 5 + 33* cells_second_row_left
                second_row_right_width = title_width + 5 + width_second_row_right
                third_row_width = title_width + 5 + 33 * cells_third_row
                forth_row_width = title_width + 5 + 33 * cells_forth_row

                coordinates.write('row-1-1 {} {} {} {} \n'.format(table_x + 40, table_y + 40, first_row_left_width, row_height))
                coordinates.write('row-1-2 {} {} {} {} \n'.format(table_x + 40 + first_row_left_width + 20, table_y + 40, first_row_right_width, row_height))
                coordinates.write('row-2-1 {} {} {} {} \n'.format(table_x + 40, table_y + 40 + row_height + 10, second_row_left_width, row_height))
                coordinates.write('row-2-2 {} {} {} {} \n'.format(table_x + 40 + second_row_left_width + 20, table_y + 40 + row_height + 10, second_row_right_width, row_height))
                coordinates.write('row-3-1 {} {} {} {} \n'.format(table_x + 40, table_y + 40 + row_height * 2 + 20, third_row_width, row_height))
                coordinates.write('row-4-1 {} {} {} {} \n'.format(table_x + 40, table_y + 40 + row_height * 3 + 30, forth_row_width, row_height))


            if style == 'multiple-choice-complex':
                coordinates.write('table-{} {} {} {} {} \n'.format(table_nr, table_x + 40,table_y + 40, width - 80, height - 80))
                cell_width = int((width - 80 - middle_region_width) / (columns_number))
                cell_height = int((height - 80 - space_between_rows * (rows_number - 1)) / (rows_number))
                for row in range(rows_number):
                    for column in range(columns_number):
                        coordinates.write('cell-{}-{}-{} {} {} {} {} \n'.format(table_nr, row, column,column * cell_width + table_x + 40, row * (cell_height + space_between_rows) + table_y + 40, cell_width, cell_height))
                    coordinates.write('cell-{}-{}-{} {} {} {} {} \n'.format(table_nr, row, columns_number, columns_number * cell_width + table_x + 40 ,row * (cell_height + space_between_rows) + table_y + 40, middle_region_width, cell_height))

            if style == 'classic':
                coordinates.write('table-{} {} {} {} {} \n'.format(table_nr, table_x + 40,table_y + 40, width - 80, height - 80))
                cell_width = int((width - 80) / (columns_number))
                cell_height = int((height - 80 - space_between_rows * (rows_number - 1)) / (rows_number))
                for row in range(rows_number):
                    for column in range(columns_number):
                        coordinates.write('cell-{}-{}-{} {} {} {} {} \n'.format(table_nr, row, column, column * cell_width + table_x + 40, row * (cell_height + space_between_rows) + table_y + 40, cell_width, cell_height))

            if style == 'multiple-choice':
                coordinates.write('table-{} {} {} {} {} \n'.format(table_nr, table_x + 40,table_y + 40, width - 80, height - 80))
                cell_width = int((width - 80) / (columns_number))
                cell_height = int((height - 80 - space_between_rows * (rows_number - 1)) / (rows_number))
                for row in range(rows_number):
                    for column in range(columns_number):
                        coordinates.write('cell-{}-{}-{} {} {} {} {} \n'.format(table_nr, row, column, column * cell_width + table_x + 40, row * (cell_height + space_between_rows) + table_y + 40, cell_width, cell_height))

            tables.append(boxA)

    # Add some random noise to the generated page
    background = add_noise(texture(blank_image(background=230), sigma=4), sigma=10)
    background = np.resize(background, (1510, 1100))
    page = np.array(page * 0.7 + background * 0.3, dtype=np.uint8)

    cv2.imwrite('table_images/table_{}.jpg'.format(image_nr), page)
    coordinates.close()



# Generate the tables using multiprocessing
try:
    os.makedirs("table_images")
except:
    pass
pool = Pool(4)
pool.map(generate_data, range(2000))