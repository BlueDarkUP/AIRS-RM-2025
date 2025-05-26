def estimate_distance(current_obj_pixel_height,
                      calib_obj_pixel_height_at_known_dist,
                      known_calib_distance_m):
    estimated_dist = known_calib_distance_m * (calib_obj_pixel_height_at_known_dist / current_obj_pixel_height)
    return estimated_dist

