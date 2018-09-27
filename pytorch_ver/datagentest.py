from data_generator import DataGenerator

data_generator = DataGenerator("branin", 1,\
        True, False, [-5, 5], [1, 1000000],\
        [1, 1000], [1, 500], 1, False, False)

(train_x, train_y),(test_x, test_y) = data_generator.get_train_test_sample()
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

test_data = np.concatenate((test_x, 
