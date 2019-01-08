# ML-LinR_LogR_JS
This is a basic machine learning library for linear and logistic regression, the fundamental hyperparameter settings (learning rate, lambda, iterations) are included. 

You can use it by copying the entire function to the source code or create your own library by modifying it.

I highly expected the room for the algorithm improvement. In case you (lucky enough) find a bug or have any feedback positive or negative, I am really looking forward to knowing it, and of course will make this work more robust and efficient!

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

Probably your accuracy is not good enough, that is because several hyperparameters are set as default, try to tune these values and you may get better accuracy. Meanwhile, you can also play with different models to increase the training efficiency. 

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
Although the definition of the hyperparameters is fully optional as the default values have been specified already, you can  define them in the function reg.LinR({}) in the code above. The default values are used as input for the following code, you can also check the source code the more details.
```js
var reg = new ML.LinR({
    iterations: 1000,
    learning_rate: 0.001,
    lambda: 0,
    print_cost: false
});
```
