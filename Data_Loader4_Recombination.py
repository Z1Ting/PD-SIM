



def data_flow4_Recombination(img_path, sample_name, sample_labels, center_cors,
              batch_size, patch_size, num_patches, shuffle_flag=True, index_list_1,index_list_2):

    margin = int(np.floor((patch_size - 1) / 2.0))  #new +1

    input_shape = (batch_size, 1, patch_size, patch_size, patch_size)
    output_shape = (batch_size, 1)

    img_path = img_path + '{}'
    n = 0



    while True:
        if shuffle_flag:
            sample_name = np.array(sample_name)
            sample_labels = np.array(sample_labels)
            permut = np.random.permutation(len(sample_name))
            np.take(sample_name, permut, out=sample_name)
            np.take(sample_labels, permut, out=sample_labels)
            sample_name = sample_name.tolist()
            sample_labels = sample_labels.tolist()

        inputs = []
        for i_input in range(num_patches):
            inputs.append(np.zeros(input_shape, dtype='float32'))
        outputs = np.ones(output_shape, dtype=np.compat.long)



        i_batch = 0

        for i_iter in range(len(sample_name)):
            # print(img_path)
            img_dir = img_path.format(sample_name[i_iter].strip())
            # print(img_dir)

            I = sitk.ReadImage(img_dir)
            img = np.array(sitk.GetArrayFromImage(I))

            for i_patch in range(center_cors.shape[0]):
                x_cor = int(center_cors[i_patch, 0])
                y_cor = int(center_cors[i_patch, 1])
                z_cor = int(center_cors[i_patch, 2])
                img_patch = img[x_cor - margin: x_cor + margin + 1,
                                y_cor - margin: y_cor + margin + 1,
                                z_cor - margin: z_cor + margin + 1]

                inputs[i_patch][i_batch, 0, :, :, :] = img_patch

            outputs[i_batch, :] = sample_labels[i_iter] * outputs[i_batch, :]

            i_batch += 1


            if i_batch == batch_size:

                inputs_1 = [inputs[i] for i in index_list_1]
                inputs_2 = [inputs[i] for i in index_list_2]

                inputs_1 = torch.from_numpy(np.array(inputs_1))
                inputs_2 = torch.from_numpy(np.array(inputs_2))
                inputs_2 = inputs_2.squeeze()

                result1 = np.zeros((batch_size, patch_size * int(np.cbrt(len(index_list_1))),
                                   patch_size * int(np.cbrt(len(index_list_1))),
                                   patch_size * int(np.cbrt(len(index_list_1)))))
                result2 = np.zeros((batch_size, patch_size * int(np.cbrt(len(index_list_1))),
                                   patch_size * int(np.cbrt(len(index_list_1))),
                                   patch_size * int(np.cbrt(len(index_list_1)))))
                arrays = [np.random.rand(21, 21, 21) for i in range(8)]





                for j in range(len(inputs_2[0])):
                    for i in range(len(inputs_2)):

                        x = i % int(np.cbrt(len(index_list_1)))
                        y = (i // int(np.cbrt(len(index_list_1)))) % int(np.cbrt(len(index_list_1)))

                        z = i // len(index_list_1)
                        result1[j][x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size,
                        z * patch_size:(z + 1) * patch_size] = inputs_1[i][j]

                        result2[j][x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size,
                        z * patch_size:(z + 1) * patch_size] = inputs_2[i][j]

                result1 = torch.from_numpy(np.array(result1))
                result2 = torch.from_numpy(np.array(result2))
                result1 = result1.unsqueeze(1).to(torch.float32)
                result2 = result2.unsqueeze(1).to(torch.float32)





                yield result1, result2,  outputs
                inputs = []
                input_shape = (batch_size, 1, patch_size, patch_size, patch_size)

                for i_input in range(num_patches):
                    inputs.append(np.zeros(input_shape, dtype='float32'))


                outputs = np.ones(output_shape, dtype=np.compat.long)
                i_batch = 0
                n += 1




def tst_data_flow_recombination(img_path, sample_name, sample_lbl, center_cors, patch_size, num_patches,index_list_1,index_list_2):
    input_shape = (1, 1, patch_size, patch_size, patch_size)
    output_shape = (1, 1)

    margin = int(np.floor((patch_size - 1) / 2.0))


    img_path = img_path + '{}'
    img_dir = img_path.format(sample_name.strip())
    I = sitk.ReadImage(img_dir)
    img = np.array(sitk.GetArrayFromImage(I))

    inputs = []
    for i_input in range(num_patches):
        inputs.append(np.zeros(input_shape, dtype='float32'))

    for i_patch in range(center_cors.shape[0]):
        x_cor = int(center_cors[i_patch, 0])
        y_cor = int(center_cors[i_patch, 1])
        z_cor = int(center_cors[i_patch, 2])

        img_patch = img[x_cor - margin: x_cor + margin + 1,
                        y_cor - margin: y_cor + margin + 1,
                        z_cor - margin: z_cor + margin + 1]

        inputs[i_patch][0, 0, :, :, :] = img_patch

    outputs = sample_lbl * np.ones(output_shape, dtype='long')

    inputs_1 = [inputs[i] for i in index_list_1]
    inputs_2 = [inputs[i] for i in index_list_2]

    inputs_1 = torch.from_numpy(np.array(inputs_1))
    inputs_2 = torch.from_numpy(np.array(inputs_2))

    inputs_2 = inputs_2.squeeze()

    result1 = np.zeros((1, patch_size * int(np.cbrt(len(index_list_1))),
                        patch_size * int(np.cbrt(len(index_list_1))),
                        patch_size * int(np.cbrt(len(index_list_1)))))
    result2 = np.zeros((1, patch_size * int(np.cbrt(len(index_list_1))),
                        patch_size * int(np.cbrt(len(index_list_1))),
                        patch_size * int(np.cbrt(len(index_list_1)))))




    for j in range(0):  # 21
        for i in range(len(inputs_2)):  #

            x = i % int(np.cbrt(len(index_list_1)))
            y = (i // int(np.cbrt(len(index_list_1)))) % int(np.cbrt(len(index_list_1)))

            z = i // len(index_list_1)
            result1[j][x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size,
            z * patch_size:(z + 1) * patch_size] = inputs_1[i][j]

            result2[j][x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size,
            z * patch_size:(z + 1) * patch_size] = inputs_2[i][j]

    result1 = torch.from_numpy(np.array(result1))
    result2 = torch.from_numpy(np.array(result2))
    result1 = result1.unsqueeze(1).to(torch.float32)
    result2 = result2.unsqueeze(1).to(torch.float32)


    return result1, result2, outputs








if __name__ == "__main__":

    train_loader = data_flow4_Recombination(img_path, samples_train, labels_train, template_cors,
                                                                batch_size, patch_size, patch_num,index_list_1,index_list_2)

    for epoch_counter in range(2):
        for i_batch in range(len(samples_valid)):
            inputs, _, outputs = tst_data_flow_recombination(img_path, samples_valid[i_batch], labels_valid[i_batch],
                                                             template_cors, patch_size, patch_num,index_list_1,index_list_2)
            print(inputs.shape)

