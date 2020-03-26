from mvnTools.Pipelines import std_2d_segment, segment_2d_to_meshes


img_enhanced, mask, img_skel, img_dist = std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/original_sample.tif',
                                                        connected=True)
verts, faces, mesh = segment_2d_to_meshes(img_dist, coarse=200, fine=20)

