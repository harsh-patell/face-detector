from keras.models import Model
import tensorflow as tf

aug_data_train_len = 4200
batches_per_epoch = aug_data_train_len
lr_decay = (1./0.75 -1)/batches_per_epoch
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)

class FaceTracker(Model):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.classloss = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        images, labels = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(images, training=True)
            
            batch_classloss = self.classloss(labels[0], classes)
            batch_lloss = self.lloss(tf.cast(labels[1], tf.float32), coords)
            
            total_loss = batch_lloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_lloss}

    def test_step(self, batch, **kwargs):
        images, labels = batch
        classes, coords = self.model(images, training=False)

        batch_classloss = self.classloss(labels[0], classes)
        batch_lloss = self.lloss(tf.cast(labels[1], tf.float32), coords)
        total_loss = batch_lloss+0.5*batch_classloss

        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_lloss}

    def call(self, images, **kwargs):
        return self.model(images, **kwargs)
