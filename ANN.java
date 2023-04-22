
import java.io.Serializable;

public class ANN implements Serializable
{
    private int numInputNode;
    private int numHiddenNode;
    private int numOutputNode;
    private double learning_rate;

    private double[][] weightsIH;
    private double[][] weightsHO;
    private double[] biasHL;
    private double[] biasOL;

    public ANN(int numInputNode, int numHiddenNode, int numOutputNode, double learning_rate){
        this.numInputNode = numInputNode;
        this.numHiddenNode = numHiddenNode;
        this.numOutputNode = numOutputNode;
        this.learning_rate = learning_rate;

        this.weightsIH = Matrix.matrix(numInputNode, numHiddenNode);
        this.weightsHO = Matrix.matrix(numHiddenNode, numOutputNode);
        this.biasHL = Matrix.vector(numHiddenNode);
        this.biasOL = Matrix.vector(numOutputNode);
    }

    public double[] feed_forward(double[] inputs){
        double[] hidden = Matrix.dot(Matrix.transpose(weightsIH), inputs);
        hidden = Matrix.sigmoid(Matrix.add(hidden, biasHL));

        double[] output = Matrix.dot(Matrix.transpose(weightsHO), hidden);
        output = Matrix.add(output, biasOL);

        return Matrix.sigmoid(output);
    }

    public double backpropagation(double[] inputs, double[] targets){
        double error = 0;

        double[] hidden = Matrix.dot(Matrix.transpose(weightsIH), inputs);
        hidden = Matrix.sigmoid(Matrix.add(hidden, biasHL));

        double[] output = Matrix.dot(Matrix.transpose(weightsHO), hidden);
        output = Matrix.sigmoid(Matrix.add(output, biasOL));

        double[] errorOutputs = Matrix.substract(targets, output);

        for(double d : errorOutputs)
            error += Math.abs(d);

        double[] deltaO =  Matrix.multiply(output, errorOutputs);
        deltaO = Matrix.multiply(deltaO, Matrix.substract(1, output));
        biasOL = Matrix.add(biasOL, Matrix.multiply(deltaO, learning_rate));

        double[] deltaH = Matrix.dot(weightsHO, deltaO);
        deltaH = Matrix.multiply(deltaH, Matrix.multiply(hidden, Matrix.substract(1, hidden)));
        biasHL = Matrix.add(biasHL, Matrix.multiply(deltaH, learning_rate));


        for (int k = 0; k < numInputNode; k++) {
            for (int l = 0; l < numHiddenNode; l++) {
                weightsIH[k][l] += learning_rate * deltaH[l] * inputs[k];
            }
        }

        for (int k = 0; k < numHiddenNode; k++) {
            for (int l = 0; l < numOutputNode; l++) {
                weightsHO[k][l] += learning_rate * deltaO[l] * hidden[k];
            }
        }
        
        return error;
    }
    
    public double[][] getWeightIH(){
        return weightsIH;
    }
}
