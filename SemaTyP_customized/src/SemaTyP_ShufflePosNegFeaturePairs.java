import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;

public class SemaTyP_ShufflePosNegFeaturePairs {

	public static void main(String [] args) {

		try {
			boolean readpos=true;
			
            File f2 = new File("/media/fot/USB-FOT/FOT/python/SemaTyP-features-shuffle.csv");  //0-16 char in 17
            FileWriter fw = new FileWriter(f2);
            BufferedWriter out = new BufferedWriter(fw);
            
            File f1 = new File("/media/fot/USB-FOT/FOT/Research/Graph analysis @dimokritos/Drug target interactions/SemaTyp/drug-gene-lc/SemaTyP-features.csv"); 
            FileReader posfr = new FileReader(f1);
            BufferedReader posbr = new BufferedReader(posfr);
            FileReader negfr = new FileReader(f1);
            BufferedReader negbr = new BufferedReader(negfr);
            
            String negline, posline;
            //reach start of each class
            String header =  posbr.readLine();
            while ((negline = negbr.readLine()) != null) {
            	String [] brk= negline.split(",");
                if (brk[481].equals("0"))
                	break;
            }
            System.out.println("First Neg line: "+negline);
            out.write(header+"\n");
            
            posline=posbr.readLine();
            String prevPosPair="";
            String prevNegPair="";
            String [] brk=posline.split(",");
            //now start mixing classes to output file
            while (brk[481].equals("1")) {
            	
            	if (readpos) {
            		out.write(posline+"\n");
            		String curPair=prevPosPair;
            		while ((posline=posbr.readLine())!=null) {
            			String []line = posline.split(",");
            			curPair=line[0];
            			if (!curPair.equals(prevPosPair)) {  //new pair
            				prevPosPair=curPair;
            				readpos=false;
            				break;
            			}
            			out.write(posline+"\n");
            		}
            	}
            	else {
            		out.write(negline+"\n");
            		String curPair=prevNegPair;
            		while ((negline=negbr.readLine())!=null) {
            			String []line = negline.split(",");
            			curPair=line[0];
            			if (!curPair.equals(prevNegPair)) {  //new pair
            				prevNegPair=curPair;
            				readpos=true;
            				break;
            			}
            			out.write(negline+"\n");
            		}		
            	}
            	brk= posline.split(",");
               
            }
            //Dont write remaining neg??
            
            posfr.close();
            posbr.close();
            negfr.close();
            negbr.close();

                 
            out.flush();
            out.close();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
}
