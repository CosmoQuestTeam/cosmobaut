import getopt
import sys
import ImageSlicer


def get_arguments():
    image_file = None
    slice_size = 450
    slice_overlap = 50
    craters_file = None
    coordinates = None
    output_directory = "output/"

    try:
        arguments, remainder = getopt.getopt(sys.argv[1:], "", ["image_file=", "slice_size=", "slice_overlap=",
                                                                "craters_file=", "output_directory=", "coordinates="])
    except getopt.GetoptError:
        print("image_slicer --image_file=<image_location>")
        sys.exit(2)

    for argument, value in arguments:
        if argument == "--image_file":
            image_file = value
        elif argument == "--slice_size":
            slice_size = value
        elif argument == "--slice_overlap":
            slice_overlap = value
        elif argument == "--craters_file":
            craters_file = value
        elif argument == "--coordinates":
            coordinates = value
        elif argument == "--output_directory":
            output_directory = value

    if image_file is None:
        print("image_slicer --image_file=<image_location>")
        sys.exit(2)

    return {
        "image_file": image_file,
        "slice_size": slice_size,
        "slice_overlap": slice_overlap,
        "craters_file": craters_file,
        "coordinates": coordinates,
        "output_directory": output_directory}


arguments = get_arguments()
image_slicer = ImageSlicer.ImageSlicer(arguments["image_file"],
                                       arguments["slice_size"],
                                       arguments["slice_overlap"],
                                       arguments["craters_file"],
                                       arguments["coordinates"])
image_slicer.completely_slice_image(arguments["output_directory"])
