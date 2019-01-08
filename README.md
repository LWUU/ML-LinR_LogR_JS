# ML-LinR_LogR_JS
This is a basic machine learning library for linear and logistic regression, the fundamental hyper-parameter settings (learning rate, lambda, iterations) are included. 

You can use it by copying the entire function to the source code or create your own library by modifying it.

I expected there is a lot of room to be improved for the algorithm. In case you (lucky enough to) find any of the bugs or have any feedback positive or negative, I am really looking forward to knowing it, and will try my best to make this work better!
### Get started by training the first model
```js
//Load the training sets (this is an example)
var train_X = train_Y();
var train_X = train_Y();

//Initialize the hyper-parameters
var reg = new ML.LinR({});

//Train the model
reg.train(train_X, train_Y)

//Show the accuracy based on the trained model
reg.predict(train_X)
```
The accuracy is calculated by sending all the train_X to the trained model, then compare with the correct Y label train_Y.

Probably your accuracy is not good enough, that is because several hyper-parameters are set as default, tuning these value and you may get better accuracy. Meanwhile, you can also play with different models to increase the running efficiency. 

### Model setting
Linear regression are realized in two approaches: gradient descent and vectorization.
```js
//gradient descent approach
var reg = new ML.LinR({});

//Vectorization approach
var reg = new ML.LinRM({});
```
Logistic regression are able to distinguish two items or multiple items.
```js
//two items
var log = new ML.LogR({});

//multiple items
var log = new ML.MLogR({});
```
### Hyper-parameters setting
Although the defination of the hyper-parameters is optional as the default values have been specified already, you can srill define them in the function reg.init({}) in the code above. The default values are used as input.
```js
var reg = new ML.LinR({
    iterations: 1000,
    learning_rate: 0.001,
    lambda: 0,
    print_cost: false
});
```
