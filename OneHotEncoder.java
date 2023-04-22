
import java.util.Arrays;
import java.io.Serializable;
import java.io.File;
import java.io.ObjectInputStream;
import java.io.FileInputStream;
import java.io.ObjectOutputStream;
import java.io.FileOutputStream;

public class OneHotEncoder implements Serializable
{
	public final static String validChars;
	public final static int onehotlength;
	public final static int epoch;
	
	static{
		validChars = "1234567890qwertyuiopasdfghjklzxcvbnm,.@#$_&-+()/*\"':;!?~`|•√π÷×¶∆£¢€¥^°={}\\%©®™✓[]<>";
		onehotlength = validChars.length();
		epoch = 55000;
	}
	
	public static double[] onehotencode(String wordortext){
		double[] vector = new double[onehotlength];
		int index;
		for(char c : (wordortext + "").toLowerCase().toCharArray()){
			index = validChars.indexOf(c);
			if(index > -1){
				vector[index] += 1 + wordortext.indexOf(c+"");
			}
		}
		return vector;
	}
	
	private final int dimension;
	private final ANN encoder_network;
	private final ANN decoder_network;
	
	public OneHotEncoder(int dimension, String corpus){
		this.dimension = dimension;
		encoder_network = new ANN(onehotlength, 15, dimension, 0.01);
		decoder_network = new ANN(dimension, 15, onehotlength, 0.03);
		train();
		trainDecoder(corpus);
	}

	public void trainDecoder(String corpus)
	{
		String[] words = corpus.split("[\\s]+");
		double[][] inputs = new double[words.length][];
		double[][] targets = new double[words.length][];
		
		for(int i=0; i<inputs.length; ++i){
			targets[i] = onehotencode(words[i]);
			inputs[i] = onehot2vector(targets[i]);
			
		}
		
		for(int e=0; e<epoch * 5; ++e){
			for(int i=0; i<inputs.length; ++i){
				double err = decoder_network.backpropagation(inputs[i], targets[i]);
				if(e % 5500 == 0)
					System.out.println("E : " + e + " | Er : " + err);
			}
		}
	}
	
	private void train(){
		double[][] sample_in = new double[6][onehotlength];
		sample_in[0][0] = 1;
		sample_in[1][onehotlength-1] = 1;
		sample_in[2][Math.round(onehotlength / 2)] = 1;
		sample_in[3][2] = 4;
		sample_in[4][onehotlength-4] = 2;
		sample_in[5][(int)Math.max(0, Math.round(Math.random() * onehotlength - 1))] = 8;
		
		double[][] sample_out = new double[6][dimension];
		for(int i=0; i<sample_out.length; ++i){
			for(int k=0; k<sample_out[i].length; ++k){
				sample_out[i][k] = 0.5;
			}
		}
		
		
		for(int i=0; i<sample_in.length; ++i){
			for(int k=0; k<sample_in[i].length; ++k){
				if(sample_in[i][k] == 0 ){
					continue;
				}
				
				double distance = (double) k / onehotlength;
				int newindex = (int)Math.round(dimension * distance);
				int finalindex = Math.max(0, Math.min(newindex, dimension - 1));
				sample_out[i][finalindex] += sample_in[i][k];
				if(finalindex> 0){
					sample_out[i][finalindex-1] += 0.5;
				}
				if(dimension > finalindex + 1){
					sample_out[i][finalindex+1] += 0.5;
				}
			}
		}
		
		for(int i=0; i<sample_out.length; ++i){
			double max = 0;
			for(int k=0; k<sample_out[i].length; ++k){
				max += sample_out[i][k];
			}
			for(int k=0; k<sample_out[i].length; ++k){
				sample_out[i][k] /= max;
			}
		}
		
		
		for(int e=0; e<epoch; ++e){
			for(int i=0; i<sample_in.length; ++i){
				double err = encoder_network.backpropagation(sample_in[i], sample_out[i]);
				if(e % 5000 == 0)
					System.out.println("E : " + e + " | Er : " + err);
			}
		}
	}
	
	public double[] text2vector(String text){
		return encoder_network.feed_forward(onehotencode(text));
	}
	
	public double[] onehot2vector(double[] onehot){
		return encoder_network.feed_forward(onehot);
	}
	
	public double[] vector2onehot(double[] vector){
		return decoder_network.feed_forward(vector);
	}
	
	public void saveToFile(File file) throws Throwable{
		ObjectOutputStream oout = null;
		try{
			oout = new ObjectOutputStream(new FileOutputStream(file));
			oout.writeObject(this);
			oout.close();
		}catch(Exception e){
			try{
				oout.close();
				throw e;
			}catch(Exception t){
				throw t;
			}
		}
	}
	
	public static OneHotEncoder fromFile(File file) throws Throwable{
		ObjectInputStream oin = null;
		try{
			oin = new ObjectInputStream(new FileInputStream(file));
			OneHotEncoder en = (OneHotEncoder) oin.readObject();
			oin.close();
			return en;
		}catch(Exception e){
			try{
				oin.close();
				throw e;
			}catch(Exception t){
				throw t;
			}
		}
	}
}
