import matplotlib.pyplot as plt

#create lists of x and y values
x = ['imagenet/500/10','coco/500/10','coco/1000/10','coco/1000/20']
y1 = [1.2359, 0.6259, 0.4382, 0.34]
y2 = [0.59859, 0.64876, 0.67424, 0.69]

# keyword arguments
plt.plot(x, y1,color='green', marker='o',
            linewidth=2, markersize=12)
plt.plot(x, y2,color='red', marker='o',
            linewidth=2, markersize=12)

plt.xlabel('model/steps/epochs')
plt.ylabel('loss & accuracy')
plt.title('Loss & accuracy')

# everything is drawn in the background, call show to see it
plt.show()
