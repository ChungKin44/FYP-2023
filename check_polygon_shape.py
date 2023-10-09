import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET

def plot_objects_from_xml(xml_content):
    # Parse XML string
    root = ET.fromstring(xml_content)

    fig, ax = plt.subplots()

    # Go through each object in the XML
    for obj in root.findall('object'):
        obj_name = obj.find('name').text

        # Extract polygon coordinates
        polygon_coords = []
        for i in range(1, 100):  # Assuming a max of 100 points; you can adjust this if needed
            x = obj.find(f'polygon/x{i}')
            y = obj.find(f'polygon/y{i}')
            if x is not None and y is not None:
                polygon_coords.append((float(x.text), float(y.text)))
                print(polygon_coords)
            else:
                break

        # Create a Polygon patch and add it to the plot
        poly_patch = patches.Polygon(polygon_coords, closed=True, edgecolor='r', fill=False)
        ax.add_patch(poly_patch)

        # Labeling the object (Optional)
        bbox_color = 'yellow'
        if obj_name == 'leaf':
            line_color = 'green'
        else:
            line_color = 'blue'
            bbox_color = 'lightblue'
        x, y = polygon_coords[0]
        ax.text(x, y, obj_name, fontsize=10, bbox=dict(facecolor=bbox_color, alpha=0.5))

    ax.set_xlim(0, int(root.find('size/width').text))
    ax.set_ylim(0, int(root.find('size/height').text))
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invert Y-axis for correct visualization
    plt.show()

# You can then call the function with the XML content string:
# plot_objects_from_xml(xml_content)

def read_xml_file(filename):
    with open(filename, 'r') as file:
        return file.read()

# Define the plot_objects_from_xml function here...

# Read the XML content from the file
xml_content = read_xml_file('C:/Users/willi/PycharmProjects/modelling/Dataset/train/0822_1153_jpg.rf.17df772e35fe0e6007cf998a70f12ab4.xml')

# Visualize the objects from the XML content
plot_objects_from_xml(xml_content)