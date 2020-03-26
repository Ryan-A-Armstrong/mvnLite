from mvnTools.Pipelines import std_2d_segment, segment_2d_to_meshes


img_enhanced, mask, img_skel, img_dist = std_2d_segment('/home/ryan/Desktop/mvn-analysis/data/original_sample.tif',
                                                        connected=True, bth_k=1, wth_k=1)
verts, faces, mesh = segment_2d_to_meshes(img_dist, img_skel, coarse=100, fine=10)

