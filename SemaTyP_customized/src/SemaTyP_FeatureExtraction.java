import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;


public class SemaTyP_FeatureExtraction {

	public static int featureLines=0;
	
	public static void extractFeaturesForPathsFile(String readFile, String outFile, String drug, String disease, HashMap <String,String[]> nodePairs_relationsOcc, String groundtruth, List<String> nodeSemTypesList, Terms terms)  throws Exception{
		
		FileWriter fw = new FileWriter(outFile, true);
		BufferedWriter bw = new BufferedWriter(fw);
		PrintWriter out = new PrintWriter(bw);
		
		
		int featureLines=0;
		//reading CUI sequence per path line
		FileReader filerdr = new FileReader(readFile);
		BufferedReader in = new BufferedReader(filerdr);
		String line=in.readLine(); //header
		while(( line = in.readLine() ) != null   ) {
			String [] nodes =  line.split(",");
			int l=nodes.length;
			String [] entities = new String[l+2];
			entities[0]=drug;
			entities[l+1]=disease;
			for (int i=1; i<l+1; i++)
				entities[i]= nodes[i-1];

			int [] objectSemOccurences= new int[SemaTyP_Neo4JAlgorithms.SEMANTIC_TYPES];
			int [] subjectSemOccurences = new int[SemaTyP_Neo4JAlgorithms.SEMANTIC_TYPES];
			int [] relationSemOccurences = new int[SemaTyP_Neo4JAlgorithms.RELATIONS];
			
			out.print(drug+"_"+disease+",");

			for (int i=0; i<l+1; i++) {
 
				String pair = entities[i]+entities[i+1];

				String [] nodePair_relationsOcc = nodePairs_relationsOcc.get(pair);
				if (nodePair_relationsOcc==null) {
					System.out.println("PROBLEM WITH PAIR: "+pair+" in line: "+line);
					continue;
				}	
				//same entities in pair - which means extendProteinNode=true;
				if(entities[i].equals(entities[i+1])) {
					nodePair_relationsOcc = nodePairs_relationsOcc.get(pair);
					for (int j=0; j<SemaTyP_Neo4JAlgorithms.SEMANTIC_TYPES; j++)
						subjectSemOccurences[j]=objectSemOccurences[j];
	 				continue;
				}
				
				for (int j=0; j<SemaTyP_Neo4JAlgorithms.SEMANTIC_TYPES; j++)
					subjectSemOccurences[j]=0;
 				for (int j=0; j<SemaTyP_Neo4JAlgorithms.RELATIONS; j++)
 					relationSemOccurences[j]=0;
				// Need to ADD previous pair objString occurences!!!! (added in pathline string)
				if (i>0) {
						
					for (int o=0; o<SemaTyP_Neo4JAlgorithms.SEMANTIC_TYPES; o++)
						subjectSemOccurences[o]+=objectSemOccurences[o];
					for (int o=0; o<SemaTyP_Neo4JAlgorithms.SEMANTIC_TYPES; o++)
						objectSemOccurences[o]=0;
				}

				for (int j=0; j<SemaTyP_Neo4JAlgorithms.NON_NORMALIZED_RELATIONS; j++) {
    	        	 
    	        	String relTypeLine=nodePair_relationsOcc[j]; 
					if ((relTypeLine!=null) &&(!relTypeLine.equals(""))) {
						String [] occurences = relTypeLine.split("-");
						String subjString = occurences[0];
						String relString = occurences[1];
							
						String [] subjStrings = subjString.split("@");
						for (String subjectType: subjStrings) {
							String []subjArray = subjectType.split("x");
							int semTypeIndex = nodeSemTypesList.indexOf(subjArray[1]);
							subjectSemOccurences[semTypeIndex]+=Integer.parseInt(subjArray[0]);
						}
						String [] relArray = relString.split("x");
						int relOcc = Integer.parseInt(relArray[0]);
						String relAbnormalType = relArray[1];
						String relType = terms.normalizedNames.get(relAbnormalType);
						int index = terms.relationTypes.indexOf(relType);
						relationSemOccurences[index] += relOcc;
						
						if ((i!=l) && (occurences.length<3)) {
							System.out.println("OOPS! PROBLEM WITH PAIR: "+pair+" in line: "+line);
							System.out.println("&&&&& no object in relTypeLine="+relTypeLine);
						}
						if (i<l) {
							
							String objString = occurences[2];
								
							String [] objStrings = objString.split("@");
							for (String objectType: objStrings) {
								String []objArray = objectType.split("x");
								int semTypeIndex = nodeSemTypesList.indexOf(objArray[1]);
								objectSemOccurences[semTypeIndex]+=Integer.parseInt(objArray[0]);
							}

						}
						
					}
				}

				for (int j=0; j<SemaTyP_Neo4JAlgorithms.SEMANTIC_TYPES; j++) 
					out.print(subjectSemOccurences[j]+",");
				for (int j=0; j<SemaTyP_Neo4JAlgorithms.RELATIONS; j++)
					out.print(relationSemOccurences[j]+",");

			}
			out.println(groundtruth);
			
			/*//DEBUG 
			if (featureLines++<8) {
				System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
				System.out.print("CUI path: "+entities[0]+"-(");
				for (int i=1; i<=l+1; i++) {
					String [] ros = nodePairs_relationsOcc.get(entities[i-1]+entities[i]);
					for (String ro:ros)
						if (ro!=null)
							System.out.print(ro+" , ");
					System.out.print(")- "+entities[i]+" (");
				}
				System.out.println("");	
			}*/
			
			
		}
		in.close();
		filerdr.close();
		out.close();
		bw.close();
		fw.close();
	}
}
