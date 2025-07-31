from cvaegan import GAN_trainer

trainer = GAN_trainer(alpha=0.1, gamma=20, lr=1e-5, split=0.17, batch_size=64, data_length=256, pre_training=True)

