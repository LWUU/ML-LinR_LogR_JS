ML = ML || {};
(function (ml) {
    //Linear regression (gradient descent)
    var LinR = function (config) {
        config = config || {};
        if (!config.iterations) {
            config.iterations = 1000;
        }
        if (!config.learning_rate) {
            config.learning_rate = 0.001;
        }
        if (!config.lambda) {
            config.lambda = 0.0;
        }
        if (!config.trace) {
            config.trace = false;
        }

        this.iterations = config.iterations;
        this.learning_rate = config.learning_rate;
        this.lambda = config.lambda;
        this.trace = config.trace;
    };
    LinR.prototype.train = function (data) {
        println("Linear regression start");
        var N = data.length, X = [], Y = [];
        this.dim = data[0].length;
        println("   Traning set initialization...");
        for (var i = 0; i < N; ++i) {
            var row = data[i];
            var x_i = [];
            var y_i = row[row.length - 1];
            x_i.push(1.0);
            for (var j = 0; j < row.length - 1; ++j) {
                x_i.push(row[j]);
            }
            Y.push(y_i);
            X.push(x_i);
        }
        this.theta = [];
        for (var d = 0; d < this.dim; ++d) {
            this.theta.push(0.0);
        }
        println("   Gradient descent...");
        for (var k = 0; k < this.iterations; ++k) {
            if (k % 1000 === 0) {
                println("       Iterations: " + k + " / " + this.iterations);
            }
            var Vx = this.gradient(X, Y, this.theta);
            for (var d = 0; d < this.dim; ++d) {
                this.theta[d] = this.theta[d] - this.learning_rate * Vx[d];
            }
            if (this.trace) {
                println('cost at iteration ' + k + ': ' + this.cost(X, Y, this.theta));
            }
        }
        println("Linrear regression finished");
        return {
            theta: this.theta,
            dim: this.dim,
            cost: this.cost(X, Y, this.theta),
            config: {
                learning_rate: this.learning_rate,
                lambda: this.lambda,
                iterations: this.iterations
            }
        };
    };
    LinR.prototype.gradient = function (X, Y, theta) {
        var N = X.length;
        var Vtheta = [];
        for (var d = 0; d < this.dim; ++d) {
            var g = 0;
            for (var i = 0; i < N; ++i) {
                var x_i = X[i];
                var y_i = Y[i];
                var predicted = this.h(x_i, theta);
                g += (predicted - y_i) * x_i[d];
            }
            g = (g + this.lambda * theta[d]) / N;
            Vtheta.push(g);
        }
        return Vtheta;
    };
    LinR.prototype.h = function (x_i, theta) {
        var predicted = 0.0;
        for (var d = 0; d < this.dim; ++d) {
            predicted += x_i[d] * theta[d];
        }
        return predicted;
    };
    LinR.prototype.cost = function (X, Y, theta) {
        var N = X.length;
        var cost = 0;
        for (var i = 0; i < N; ++i) {
            var x_i = X[i];
            var predicted = this.h(x_i, theta);
            cost += (predicted - Y[i]) * (predicted - Y[i]);
        }

        for (var d = 0; d < this.dim; ++d) {
            cost += this.lambda * theta[d] * theta[d];
        }

        return cost / (2.0 * N);
    };
    LinR.prototype.predict = function (X) {
        if (X[0].length) { // x is a matrix            
            var predicted_array = [];
            for (var i = 0; i < X.length; ++i) {
                var predicted = this.predict(X[i]);
                predicted_array.push(predicted);
            }
            return predicted_array;
        }
        // x is a row vector
        var x_i = [];
        x_i.push(1.0);
        for (var j = 0; j < X.length; ++j) {
            x_i.push(X[j]);
        }
        return this.h(x_i, this.theta);
    };
    ml.LinR = LinR;
    //Linear regression (vectorization)
    var LinRM = function (config) {
        config = config || {};
        if (!config.lambda) {
            config.lambda = 0.0;
        }
        if (!config.sigDig) {
            config.sigDig = 5;
        }
        this.lambda = config.lambda;
        this.sigDig = config.sigDig;
    };
    LinRM.prototype.train = function (data) {
        //println("Linear regression start");
        var N = data.length, X = [], Y = [];
        this.dim = data[0].length;
        //println("   Traning set initialization...");
        for (var i = 0; i < N; ++i) {
            var row = data[i];
            var x_i = [];
            var y_i = row[row.length - 1];
            x_i.push(1.0);
            for (var j = 0; j < row.length - 1; ++j) {
                x_i.push(row[j]);
            }
            Y.push(y_i);
            X.push(x_i);
        }
        this.theta = [];
        // First define the matrices B: xT*y and P: xT*x
        var B = new Array();
        var P = new Array();
        for (i = 0; i < this.dim; i++) {
            var sum = 0;
            for (var k = 0; k < N; k++) {
                sum = sum + X[k][i] * Y[k];
            }
            B[i] = sum;
            P[i] = new Array();
            for (var j = 0; j < this.dim; j++) {
                sum = 0;
                for (k = 0; k < N; k++) {
                    sum = sum + X[k][i] * X[k][j];
                }
                P[i][j] = sum;
            }
        }
        for (i = 1; i < this.dim; i++) {
            P[i][i] = P[i][i] + this.lambda;
        }
        //println("   Traning set calculation...");
        var invP = this.inv(P);
        for (k = 0; k < this.dim; k++) {
            sum = 0;
            for (j = 0; j < this.dim; j++) {
                sum = sum + invP[k][j] * B[j];
            }
            this.theta.push(this.roundSigDig(sum, this.sigDig));
        }
        //println("Linrear regression finished");
        return {
            theta: this.theta,
            dim: this.dim,
            cost: this.cost(X, Y, this.theta),
            config: {
                lambda: this.lambda,
                sigDig: this.sigDig
            }
        };
    };
    LinRM.prototype.inv = function (A) {
        var Length = A.length;
        var B = new Array();
        for (var i = 0; i < Length; i++) {
            B[i] = new Array();
        }
        var d = this.det(A);
        if (d === 0)
            throw "Error: singular matrix has been created, the provided training set is not valid";
        else {
            var i;
            var j;
            for (var i = 0; i < Length; i++) {
                for (var j = 0; j < Length; j++) {
                    // create the minor
                    var minor = new Array();
                    for (var k = 0; k < Length - 1; k++) {
                        minor[k] = new Array();
                    }
                    var m;
                    var n;
                    var theColumn;
                    var theRow;
                    // columns
                    for (var m = 0; m < Length - 1; m++) {
                        if (m < j)
                            theColumn = m;
                        else
                            theColumn = m + 1;
                        for (var n = 0; n < Length - 1; n++) {
                            if (n < i)
                                theRow = n;
                            else
                                theRow = n + 1;
                            minor[n][m] = A[theRow][theColumn];
                        } // n
                    } // m
                    // inverse entry
                    var temp = (i + j) / 2;
                    if (temp === Math.round(temp))
                        var factor = 1;
                    else
                        factor = -1;
                    B[j][i] = this.det(minor) * factor / d;
                }
            }
        }

        return(B);
    };
    LinRM.prototype.det = function (A) {
        var Length = A.length;
        // formal length of a matrix is one bigger
        if (Length === 1)
            return (A[0][0]);
        else {
            var i;
            var sum = 0;
            var factor = 1;
            for (var i = 0; i < Length; i++) {
                if (A[0][i] !== 0) {
                    // create the minor
                    var minor = new Array();
                    for (var k = 0; k < Length - 1; k++) {
                        minor[k] = new Array();
                    }
                    for (var m = 0; m < Length - 1; m++) {
                        if (m < i)
                            var theColumn = m;
                        else
                            theColumn = m + 1;
                        for (var n = 0; n < Length - 1; n++) {
                            minor[n][m] = A[n + 1][theColumn];
                        }
                    }
                    // compute its determinant
                    sum = sum + A[0][i] * factor * this.det(minor);
                }
                // alternating sum
                factor = -factor;
            }
        }
        return(sum);
    };
    LinRM.prototype.shiftR = function (theNumber, k) {
        if (k === 0)
            return (theNumber);
        else {
            var k2 = 1;
            var num = k;
            if (num < 0)
                num = -num;
            for (var i = 0; i < num; i++) {
                k2 = k2 * 10;
            }
        }
        if (k > 0) {
            return(k2 * theNumber);
        } else {
            return(theNumber / k2);
        }
    };
    LinRM.prototype.roundSigDig = function (theNumber, numDigits) {
        with (Math) {
            if (theNumber === 0)
                return(0);
            else if (abs(theNumber) < 0.0000000001)
                return(0);
            // warning: ignores numbers less than 10^(-12)
            else {
                var k = floor(log(abs(theNumber)) / log(10)) - numDigits;
                var k2 = this.shiftR(round(this.shiftR(abs(theNumber), -k)), k);
                if (theNumber > 0)
                    return(k2);
                else
                    return(-k2);
            }
        }
    };
    LinRM.prototype.h = function (x_i, theta) {
        var predicted = 0.0;
        for (var d = 0; d < this.dim; ++d) {
            predicted += x_i[d] * theta[d];
        }
        return predicted;
    };
    LinRM.prototype.cost = function (X, Y, theta) {
        var N = X.length;
        var cost = 0;
        for (var i = 0; i < N; ++i) {
            var x_i = X[i];
            var predicted = this.h(x_i, theta);
            cost += (predicted - Y[i]) * (predicted - Y[i]);
        }

        for (var d = 0; d < this.dim; ++d) {
            cost += this.lambda * theta[d] * theta[d];
        }

        return cost / (2.0 * N);
    };
    LinRM.prototype.predict = function (X) {
        if (X[0].length) { // x is a matrix            
            var predicted_array = [];
            for (var i = 0; i < X.length; ++i) {
                var predicted = this.predict(X[i]);
                predicted_array.push(predicted);
            }
            return predicted_array;
        }
        // x is a row vector
        var x_i = [];
        x_i.push(1.0);
        for (var j = 0; j < X.length; ++j) {
            x_i.push(X[j]);
        }
        return this.h(x_i, this.theta);
    };
    ml.LinRM = LinRM;
    //Logistic regression
    var LogR = function (config) {
        var config = config || {};
        if (!config.learning_rate) {
            config.learning_rate = 0.001;
        }
        if (!config.iterations) {
            config.iterations = 100;
        }
        if (!config.lambda) {
            config.lambda = 0;
        }
        this.learning_rate = config.learning_rate;
        this.lambda = config.lambda;
        this.iterations = config.iterations;
    };
    LogR.prototype.train = function (data) {
        println("Logistic regression start");
        println("   Training sets initialization...");
        this.dim = data[0].length;
        var N = data.length;
        var X = [];
        var Y = [];
        for (var i = 0; i < N; ++i) {
            var row = data[i];
            var x_i = [];
            var y_i = row[row.length - 1];
            x_i.push(1.0);
            for (var j = 0; j < row.length - 1; ++j) {
                x_i.push(row[j]);
            }
            X.push(x_i);
            Y.push(y_i);
        }
        this.theta = [];
        for (var d = 0; d < this.dim; ++d) {
            this.theta.push(0.0);
        }
        println("   Gradient descent...");
        for (var iter = 0; iter < this.iterations; ++iter) {
            if (iter % 1000 === 0) {
                println("       Iterations: " + iter + " / " + this.iterations);
            }
            var theta_delta = this.gradient(X, Y, this.theta);
            for (var d = 0; d < this.dim; ++d) {
                this.theta[d] = this.theta[d] - this.learning_rate * theta_delta[d];
            }
        }
        println("   Threshold computation...");
        this.threshold = this.computeThreshold(X, Y);
        println("Logistic regression finished");
        return {
            theta: this.theta,
            threshold: this.threshold,
            cost: this.cost(X, Y, this.theta),
            config: {
                learning_rate: this.learning_rate,
                lambda: this.lambda,
                iterations: this.iterations
            }
        };
    };
    LogR.prototype.computeThreshold = function (X, Y) {
        var threshold = 1.0, N = X.length;
        for (var i = 0; i < N; ++i) {
            var prob = this.predict(X[i]);
            if (Y[i] === 1 && threshold > prob) {
                threshold = prob;
            }
        }
        return threshold;
    };
    LogR.prototype.gradient = function (X, Y, theta) {
        var N = X.length;
        var Vx = [];
        for (var d = 0; d < this.dim; ++d) {
            var sum = 0.0;
            for (var i = 0; i < N; ++i) {
                var x_i = X[i];
                var predicted = this.h(x_i, theta);
                sum += ((predicted - Y[i]) * x_i[d] + this.lambda * theta[d]) / N;
            }
            Vx.push(sum);
        }
        return Vx;
    };
    LogR.prototype.h = function (x_i, theta) {
        var gx = 0.0;
        for (var d = 0; d < this.dim; ++d) {
            gx += theta[d] * x_i[d];
        }
        return 1.0 / (1.0 + Math.exp(-gx));
    };
    LogR.prototype.predict = function (X) {
        if (X[0].length) { // x is a matrix            
            var predicted_array = [];
            for (var i = 0; i < X.length; ++i) {
                var predicted = this.predict(X[i]);
                predicted_array.push(predicted);
            }
            return predicted_array;
        }

        var x_i = [];
        x_i.push(1.0);
        for (var j = 0; j < X.length; ++j) {
            x_i.push(X[j]);
        }
        return this.h(x_i, this.theta);
    };
    LogR.prototype.cost = function (X, Y, theta) {
        var N = X.length;
        var sum = 0;
        for (var i = 0; i < N; ++i) {
            var y_i = Y[i];
            var x_i = X[i];
            sum += -(y_i * Math.log(this.h(x_i, theta)) + (1 - y_i) * Math.log(1 - this.h(x_i, theta))) / N;
        }

        for (var d = 0; d < this.dim; ++d) {
            sum += (this.lambda * theta[d] * theta[d]) / (2.0 * N);
        }
        return sum;
    };
    //Multiclass logistic regression
    ml.LogR = LogR;
    var MLogR = function (config) {
        var config = config || {};
        if (!config.learning_rate) {
            config.learning_rate = 0.001;
        }
        if (!config.iterations) {
            config.iterations = 100;
        }
        if (!config.lambda) {
            config.lambda = 0;
        }
        this.learning_rate = config.learning_rate;
        this.lambda = config.lambda;
        this.iterations = config.iterations;
    };
    MLogR.prototype.train = function (data, classes) {
        println("Multiclass logistic regression start");
        this.dim = data[0].length;
        var N = data.length;
        println("   Multiclass value identification...");
        if (!classes) {
            classes = [];
            for (var i = 0; i < N; ++i) {
                var found = false;
                var label = data[i][this.dim - 1];
                for (var j = 0; j < classes.length; ++j) {
                    if (label === classes[j]) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    classes.push(label);
                }
            }
        }
        this.classes = classes;
        this.logistics = {};
        var result = {};
        for (var k = 0; k < this.classes.length; ++k) {
            var c = this.classes[k];
            this.logistics[c] = new ml.LogR({
                learning_rate: this.learning_rate,
                lambda: this.lambda,
                iterations: this.iterations
            });
            var data_c = [];
            for (var i = 0; i < N; ++i) {
                var row = [];
                for (var j = 0; j < this.dim - 1; ++j) {
                    row.push(data[i][j]);
                }
                row.push(data[i][this.dim - 1] === c ? 1 : 0);
                data_c.push(row);
            }
            result[c] = this.logistics[c].fit(data_c);
        }
        println("Multiclass logistic regression finished");
        return result;
    };
    MLogR.prototype.predict = function (X) {
        if (X[0].length) { // x is a matrix            
            var predicted_array = [];
            for (var i = 0; i < X.length; ++i) {
                var predicted = this.predict(X[i]);
                predicted_array.push(predicted);
            }
            return predicted_array;
        }
        var max_prob = 0.0;
        var best_c = '';
        for (var k = 0; k < this.classes.length; ++k) {
            var c = this.classes[k];
            var prob_c = this.logistics[c].transform(X);
            if (max_prob < prob_c) {
                max_prob = prob_c;
                best_c = c;
            }
        }
        return best_c;
    };
    ml.MLogR = MLogR;
})(ML);