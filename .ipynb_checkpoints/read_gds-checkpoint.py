import gdspy


def extra_labels_and_polygons(gds_path):
    gds_file = gdspy.GdsLibrary(infile=gds_path)
    top_cell = gds_file.top_level()[0]
    labels = top_cell.labels
    polygons = top_cell.polygons

    return  labels, polygons


if __name__ == "__main__":
    file_path = ("/media/kael/台电/learn_more_from_life/computer/EDA/"
                 "work/prj/rc_predict/data/pattern1/pattern_results/"
                 "pattern_TLineEndHat_M1M2__0d5_0d3/pattern_TLineEndHat_M1M2__0d5_0d3.gds")
    gds_file = gdspy.GdsLibrary(infile=file_path)
    top_cell = gds_file.top_level()[0]
    labels = top_cell.labels
    polygons = top_cell.polygons
