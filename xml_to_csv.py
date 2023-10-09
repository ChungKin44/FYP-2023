from main import parse_xml, convert_label


def xml_to_csv(root_dir):  # XML to CSV format
    bbox = []
    labels = []
    for xml_file in glob.glob(root_dir + '/*.xml'):
        if os.path.exists(xml_file):
            bboxes, lbls, width, height = parse_xml(xml_file)
            bbox.append(bboxes)
            labels.append(lbls)
            label_in_num = convert_label(lbls)

    print(len(bbox))
    print(len(labels))
    csv_file = os.path.join(root_dir, 'annotations.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'width', 'height', 'label', 'xmin', 'ymin', 'xmax', 'ymax'])

        for i in range(len(bbox)):
            for j in range(len(bbox[i])):
                writer.writerow([os.path.basename(xml_file), width, height, labels[i][j], bbox[i][j][0], bbox[i][j][1],
                                 bbox[i][j][2], bbox[i][j][3]])

    print(f"CSV file saved at: {csv_file}")

