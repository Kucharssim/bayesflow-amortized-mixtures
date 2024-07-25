from amortizer import trainer

print("Starting training")

h = trainer.train_online(epochs=3, iterations_per_epoch=10, batch_size=512)

print("Training complete")

