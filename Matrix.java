import java.util.Arrays;
import java.util.Random;

public class Matrix
{
    private static Random random = new Random();

    

    public static double[] softmax(double[] x) {
        double total = 0;
        for(double e : x){
            total += e;
        }
        for(int i=0; i<x.length; ++i){
            x[i] /= total;
        }
        return x;
    }
    
    public static double[][] matrix(int row, int col){
        double[][] a = new double[row][col];
        for(int i=0; i<row; ++i)
            for(int k=0; k<col; ++k)
                a[i][k] = random.nextDouble();
        return a;
    }

    public static double[] vector(int size){
        double[] a = new double[size];
        for(int i=0; i<size; ++i)
            a[i] = random.nextDouble();
        return a;
    }

    public static double[] sigmoid(double[] a){
        double[] result = new double[a.length];
        for(int i=0; i<a.length; ++i)
            result[i] = sigmoid(a[i]);
        return result;
    }

    public static double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }

    public static double[] dot(double[][] a, double[] b){
        double[] result = new double[a.length];
        for(int i=0; i<a.length; ++i){
            for(int k=0; k<b.length; ++k){
                result[i] += a[i][k] * b[k];
            }
        }
        return result;
    }

    public static double[] add(double[] a, double[] b){
        if(a.length != b.length)
            throw new IllegalArgumentException("Mismatched arrays");
        double[] result = new double[a.length];

        for(int i=0; i<a.length; ++i){
            result[i] = a[i] + b[i];
        }
        return result;
    }

    public static double[][] add(double[][] a, double[][] b){
        if(a.length != b.length || a[0].length != b[0].length)
            throw new IllegalArgumentException("Mismatched arrays");
        double[][] result = new double[a.length][b[0].length];

        for(int i=0; i<a.length; ++i){
            for(int k=0; k<a[i].length; ++k){
                result[i][k] = a[i][k] + b[i][k];
            }
        }
        return result;
    }

    public static double[][] add(double[][] a, double[] b)
    {
        double[][] result = new double[a.length][a[0].length];

        for(int i=0; i<a.length; ++i){
            for(int k=0; k<a[i].length; ++k){
                result[i][k] = a[i][k] + b[i];
            }
        }
        return result;
    }

    public static double[] substract(double a, double[] b){
        double[] result = new double[b.length];

        for(int i=0; i<b.length; ++i){
            result[i] = a - b[i];
        }
        return result;
    }

    public static double[] substract(double[] a, double[] b){
        if(a.length != b.length)
            throw new IllegalArgumentException("Mismatched arrays");
        double[] result = new double[a.length];

        for(int i=0; i<a.length; ++i){
            result[i] = a[i] - b[i];
        }
        return result;
    }

    public static double[] multiply(double[] a, double[] b){
        if(a.length != b.length)
            throw new IllegalArgumentException("Mismatched arrays");
        double[] result = new double[a.length];

        for(int i=0; i<a.length; ++i){
            result[i] = a[i] * b[i];
        }
        return result;
    }

    public static double[][] multiply(double[][] a, double[] b){
        double[][] result = new double[a.length][a[0].length];

        for(int i=0; i<a.length; ++i){
            for(int k=0; k<a[0].length; ++k){
                result[i][k] = a[i][k] * b[i];
            }
        }
        return result;
    }

    public static double[][] multiply(double[][] a, double b){
        double[][] result = new double[a.length][a[0].length];

        for(int i=0; i<a.length; ++i){
            for(int k=0; k<a[0].length; ++k){
                result[i][k] = a[i][k] * b;
            }
        }
        return result;
    }

    public static double[] multiply(double[] a, double b){
        double[] result = new double[a.length];
        for(int i=0; i<a.length; ++i){
            result[i] = a[i] * b;
        }
        return result;
    }

    public static double[] divide(double[] a, double b){
        double[] result = new double[a.length];

        for(int i=0; i<a.length; ++i){
            result[i] = a[i] / b;
        }
        return result;
    }


    public static double[][] transpose(double[][] a) {
        int rows = a.length;
        int cols = a[0].length;
        double[][] result = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = a[i][j];
            }
        }
        return result;
    }
}
