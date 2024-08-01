from amortizer import trainer

print("Starting training")

h = trainer.train_online(epochs=200, iterations_per_epoch=1000, batch_size=512)

print("Training complete")

