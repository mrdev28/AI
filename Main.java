import java.io.File;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.net.URL;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main
{
	
	
	public static void main(String[] args){
		File f = new File("/");
		String corpus = "tum ho kahan, kyun mujhe hai khona";
        OneHotEncoder encoder = null;
		try{
			encoder = OneHotEncoder.fromFile(f);
			System.out.println("Using Cached One Hot Encoder");
		}catch(Throwable e){
			System.out.println(e);
			encoder = new OneHotEncoder(8, corpus);
			try
			{
				encoder.saveToFile(f);
				System.out.println("One Hot Encoder cached");
			}
			catch (Throwable t)
			{
				System.out.println(t);
				System.out.println("One Hot Encoder State not cached");
			}
		}
		
		encoder.trainDecoder(corpus);
		
		String[] words = {"tum", "hai", "khona"};
		
		for(String word : words){
			double[] original = encoder.onehotencode(word);
			double[] encoded = encoder.onehot2vector(original);
			double[] decoded = encoder.vector2onehot(encoded);
			
			System.out.println();
			System.out.println("Original : " + Arrays.toString(original));
			System.out.println("Encoded : " + Arrays.toString(encoded));
			System.out.println("Decoded : " + Arrays.toString(decoded));
			for(int i=0; i<decoded.length; ++i){
				decoded[i] = (Math.round(decoded[i]));
			}
			
			System.out.println("Decoded : " + Arrays.toString(decoded));
			
		}
		
		
		
		System.out.println("Done");
	}
    
}
