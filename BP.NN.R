########################################################################################
##
##  Backpropagation partial differential equation
##  dE/dw = dE/dout * dout/dnet * dnet/dw
##  For output neurons:
##  dE/dw = (output-actual) * 
##          differential of output activation function * 
##          input for weight
##
##  For hidden layer:
##  save 'dout/dnet * dnet/dw' for output layers
##  dE/dw = (dout/dnet * dnet/dw) * output weight *
##          differential of output function (using actual saved output) *
##          input for weight
##
########################################################################################

  ####################################### TODO #######################################
  ##  Generalize weight and input/output matrices for any size + bias
  ##  Ensure the network scales for any number of inputs/outputs
  ##  Expand the set of activation functions avaiable
  ##  Include test/train/validate splits (double cross validation)
  ##  Incorporate bootstrap aggregation of multiple networks

x<-seq(0,2*pi,0.01)
y<-sin(x)*x

x_<-mapValues(x,0,1)
y_<-mapValues(y,0,1)

input.full<-matrix(0,nrow=length(x_),ncol=2)
input.full[,1]<-x_
input.full[,2]<-1
## HACK
nets.plus.bias<-2
## HACK
output.neuron.weights<-matrix(rnorm(nets.plus.bias,0,0.2),nrow=nets.plus.bias)
hidden.neurons.weights<-matrix(rnorm(nets.plus.bias*ncol(input.full),0,0.2),nrow=ncol(input.full))
hidden.neurons.weights[,nets.plus.bias]<-c(0,100)

print(paste("Iteration:", 0, " ",testNet()))
for(p in 1:1000) {
  for(i in 1:10) {
    a<-sample(nrow(input.full),20,replace=FALSE)
    input<-input.full[a,,drop=FALSE]
    
    hidden.neurons.ouput<-input%*%hidden.neurons.weights
    hidden.neurons.ouput<-1/(1+exp(-hidden.neurons.ouput))
    output<-hidden.neurons.ouput%*%output.neuron.weights
    
    sig<-(output-y_[a])
    
    delta.E.delta.w<-crossprod(hidden.neurons.ouput,sig)/nrow(input)
    sig.hidden<-hidden.neurons.ouput*(1-hidden.neurons.ouput)*(sig%*%t(output.neuron.weights))/nrow(input)
    
    output.neuron.weights<-output.neuron.weights-1*delta.E.delta.w
    hidden.neurons.weights<-hidden.neurons.weights-1*t(crossprod(sig.hidden,input))
  }
  print(paste("Iteration:", p*10, " ",testNet()))
}


########################################################################################
##
##  Some functions used in the above code
##
########################################################################################
testNet<-function() {
  input<-input.full
  hidden.neurons.ouput<-input%*%hidden.neurons.weights
  hidden.neurons.ouput<-1/(1+exp(-hidden.neurons.ouput))
  output<-hidden.neurons.ouput%*%output.neuron.weights

  mse<-(output-y_)^2
  return(sum(mse))
}

mapValues<-function(x, from, to) {
  x<-x-min(x)
  x<-x/max(x)
  x<-x*(to-from)
  x<-x+from
  return(x)
  #return((x-min(x))/max(x-min(x))*(to-from)+from)
}

