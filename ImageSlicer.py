from PIL import Image
import os
import math
import json


class ImageSlicer:
    def __init__(self, image_file, slice_size, slice_overlap, craters_file, coordinate_text):
        self.image_file = image_file
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self.craters_file = craters_file
        coordinate_values = coordinate_text.split(",")
        self.coordinates = self.Rectangle(int(coordinate_values[0]), int(coordinate_values[1]),
                                          int(coordinate_values[2]) - int(coordinate_values[0]),
                                          int(coordinate_values[3]) - int(coordinate_values[1]))

    def read_in_crater_data(self, craters_file, image_width, image_height):
        lat_long_craters = []

        with open(craters_file) as file:
            read_first_line = False
            for line in file:
                if read_first_line:
                    try:
                        values = line.split(",")
                        lat_long_craters.append({"latitude": float(values[1]),
                                                 "longitude": float(values[2]),
                                                 "diameter": float(values[3]) / 30.32})  # 30.32 converts km to deg
                    except:
                        pass
                read_first_line = True

        image_bounds = self.Rectangle(0, 0, image_width, image_height)
        return self.get_crater_bounds(lat_long_craters, image_bounds, self.coordinates)

    def completely_slice_image(self, output_directory):
        Image.MAX_IMAGE_PIXELS = None
        image = Image.open(self.image_file)
        image_width, image_height = image.size
        all_sub_image_data = []

        crater_bounds = []
        if self.craters_file is not None:
            all_crater_bounds = self.read_in_crater_data(self.craters_file, image_width, image_height)
            for crater in all_crater_bounds:
                if 0 < crater["x"] < image_width and 0 < crater["y"] < image_height:
                    crater_bounds.append(crater)

        slice_scale = 1
        # Repeatedly cut the image height and width in half and slice the image each time
        while self.slice_size < image_width and self.slice_size < image_height:
            top = 0
            y_done = False
            while top < image_height and y_done is False:
                if top + self.slice_size >= image_height:
                    top = image_height - self.slice_size
                    y_done = True
                left = 0
                x_done = False
                while left < image_width and x_done is False:
                    if left + self.slice_size >= image_width:
                        left = image_width - self.slice_size
                        x_done = True
                    bounding_box = self.Rectangle(left, top, self.slice_size, self.slice_size)
                    all_sub_image_data.append(self.make_sub_image(image, bounding_box, self.slice_size,
                                                                  output_directory, crater_bounds, slice_scale))
                    left += self.slice_size - self.slice_overlap
                top += self.slice_size - self.slice_overlap
            image = image.resize((int(image_width / 2), int(image_height / 2)))
            for crater in crater_bounds:
                crater["x"] /= 2
                crater["y"] /= 2
                crater["width"] /= 2
                crater["height"] /= 2
            image_width, image_height = image.size
            slice_scale *= 2
        text = json.dumps(all_sub_image_data, indent=4)
        file = open(os.path.join(output_directory, "sub_image_data.json"), "w")
        file.write(text)
        file.close()

    def make_sub_image(self, image, bounding_box, output_size, output_directory, craters, slice_scale):
        local_file_location = os.path.join(output_directory, str(slice_scale),
                                           str(bounding_box.left) + "_" + str(bounding_box.top) + ".png")
        local_directory = os.path.dirname(os.path.realpath(local_file_location))

        if not os.path.exists(local_directory):
            os.makedirs(local_directory)

        image.crop((bounding_box.left, bounding_box.top, bounding_box.right, bounding_box.bottom)).resize(
            (output_size, output_size), Image.ANTIALIAS).save(local_file_location, "png", quality=100)

        sub_image_data = {"file_location": local_file_location, "left": bounding_box.left, "top": bounding_box.top,
                          "width": bounding_box.width, "height": bounding_box.height}
        if craters is not None:
            sub_image_data["craters"] = self.get_crater_data_for_region(bounding_box, craters)

        return sub_image_data

    def get_crater_data_for_region(self, bounding_box, craters):
        sub_image_craters = []
        for crater in craters:
            if (
                    bounding_box.left < crater["x"] < bounding_box.right and
                    bounding_box.top < crater["y"] < bounding_box.bottom
                    and 10 < crater["width"] < bounding_box.width * 3/4
                    and 10 < crater["height"] < bounding_box.height * 3/4):
                new_crater = {"left": max(0, round(crater["x"] - bounding_box.left - crater["width"] / 2, 1)),
                              "top": max(0, round(crater["y"] - bounding_box.top - crater["height"] / 2, 1)),
                              "right": min(bounding_box.width, round(crater["x"] - bounding_box.left + crater["width"] / 2, 1)),
                              "bottom": min(bounding_box.height, round(crater["y"] - bounding_box.top + crater["height"] / 2, 1))}
                sub_image_craters.append(new_crater)
        return sub_image_craters

    def get_crater_bounds(self, lat_long_craters, image_bounds, coordinate_bounds):
        is_orthographic = True
        all_crater_bounds = []
        lat_long_craters = sorted(lat_long_craters, key=lambda crater: crater["diameter"], reverse=True)
        for crater in lat_long_craters:
            crater_bounds = {
                "x": (crater["longitude"] - coordinate_bounds.left) / coordinate_bounds.width * image_bounds.width,
                "y": (crater["latitude"] - coordinate_bounds.top) / coordinate_bounds.height * image_bounds.height,
                "width": abs(crater["diameter"] / coordinate_bounds.width * image_bounds.width),
                "height": abs(crater["diameter"] / coordinate_bounds.height * image_bounds.height)
            }
            if is_orthographic:
                crater_bounds["width"] /= math.cos(crater["latitude"] * math.pi / 180)
            all_crater_bounds.append(crater_bounds)
        return all_crater_bounds

    class Rectangle:
        def __init__(self, left, top, width, height):
            self.left = left
            self.top = top
            self.width = width
            self.height = height
            self.right = left + width
            self.bottom = top + height
