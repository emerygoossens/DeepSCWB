

good_imgs = X_test[good_test_ind][:,:,7:57,0]

good_seg_pred = y_pred_seg_reshape[good_test_ind][:,:,7:57]


output_mat = np.zeros((good_seg_pred.shape[0],3))
for j in range(good_seg_pred.shape[0]):
    an_image = good_imgs[j,:,:]
    good_pixels = np.where(good_seg_pred[j,:,:]>.5)
    background_pixels = np.where(good_seg_pred[j,:,:]<.5)
    background_median = np.median(an_image[background_pixels])
    background_mean = np.mean(an_image[background_pixels])
    protein_values = (an_image[good_pixels] - background_median)
    protein_values[protein_values <0] = 0
    protein_sum = protein_values.sum()
    output_mat[j,0] = protein_sum
    output_mat[j,1] = background_median
    output_mat[j,2] = background_mean

protein_expression = protein_sum - good_pixels[0].shape[0]*background_median