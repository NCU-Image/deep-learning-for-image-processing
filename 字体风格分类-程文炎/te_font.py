import os
from keras.models import load_model
from keras.models import Model
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import operator
from keras import backend as K
batch_size = 32
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# model = load_model('../logs/categories_crossentroy  1_1_0.5/font_train_19660/font_resnet.h5', custom_objects={'contrastive_loss':contrastive_loss})
model = load_model('../logs/categories_crossentroy  1_1_0.5/font_train_31900/font_sia_vgg16.h5')
single_model = Model(inputs=model.layers[0].input, outputs=[model.layers[6].output])
model = single_model
# plot_model(model, to_file='single_model.png')
test_datagen = ImageDataGenerator(rescale=1./255,)
valid_generator = test_datagen.flow_from_directory(
    'E:\PycharmProjects\AI\Dataset\Fote Dataset\Test',
    target_size=(96, 96),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)
print(valid_generator.class_indices)

test_path = 'E:\PycharmProjects\AI\Dataset\Fote Dataset\Test'
# sorted() 函数对所有可迭代的对象进行排序操作。
# reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）。
label_idxs = sorted(valid_generator.class_indices.items(), key=operator.itemgetter(1))
print(label_idxs[0])
test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(96, 96),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


loss_1,acc = model.evaluate_generator(test_generator)
print(loss_1,acc)
