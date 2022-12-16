import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;


public class FeatureExtraction {

	public static int featureLines=0;
	
	public static void extractFeaturesForPathsFile(String readFile, String outFile, String drug, String target, HashMap <String,String[]> nodePairs_relationsOcc, String groundtruth, List<String> nodeSemTypesList, Terms terms)  throws Exception{
		
		final int offset = Enriched_DTI_BLKG.SEMANTIC_TYPES+Enriched_DTI_BLKG.RELATIONS;
		int [] pairline = new int[Enriched_DTI_BLKG.LENGTH*offset];
		//Initialize all occurence values in initial pairline with '0's
		for (int i=0; i<Enriched_DTI_BLKG.LENGTH; i++)
			for (int j=0; j<offset; j++) 
				pairline[i*offset+j]=0;
		
		//reading CUI sequence per path line
		FileReader filerdr = new FileReader(readFile);
		BufferedReader in = new BufferedReader(filerdr);
		String line="";
		while(( line = in.readLine() ) != null   ) {
			String [] nodes =  line.split(",");
			int l=nodes.length;
			String [] entities = new String[l+2];
			entities[0]=drug;
			entities[l+1]=target;
			for (int i=1; i<l+1; i++)
				entities[i]= nodes[i-1];

			int [] objectSemOccurences= new int[Enriched_DTI_BLKG.SEMANTIC_TYPES];
			int [] subjectSemOccurences = new int[Enriched_DTI_BLKG.SEMANTIC_TYPES];
			int [] relationSemOccurences = new int[Enriched_DTI_BLKG.RELATIONS];
			

			for (int i=0; i<l+1; i++) {
 
				String pair = entities[i]+entities[i+1];

				String [] nodePair_relationsOcc = nodePairs_relationsOcc.get(pair);
				if (nodePair_relationsOcc==null) 
					continue;
				//put NULL in order to COUNT in occurences every relation between two nodes JUST ONCE
				nodePairs_relationsOcc.put(pair,null);
				
				/*/debug
				System.out.println("PAIR: "+pair+", relation occs:");
				for (String s:nodePair_relationsOcc)
					if ((s!=null) &&(!s.equals("")))
						System.out.println("nodePair_relationsOcc:"+s);
				*/
				
			
				for (int j=0; j<Enriched_DTI_BLKG.SEMANTIC_TYPES; j++)
					subjectSemOccurences[j]=0;
 				for (int j=0; j<Enriched_DTI_BLKG.RELATIONS; j++)
 					relationSemOccurences[j]=0;

				//same entities in pair - which means extendProteinNode=true;
				if(entities[i].equals(entities[i+1])) {
					//FOT CHANGE!!!!!
					//nodePair_relationsOcc = nodePairs_relationsOcc.get(pair);
					int index = terms.relationTypes.indexOf("same_as");
					relationSemOccurences[index]=1;
		
					for (int j=0; j<Enriched_DTI_BLKG.SEMANTIC_TYPES; j++)
						subjectSemOccurences[j]=objectSemOccurences[j];
	 				//continue;
				}
				else {
					
	 				// Need to ADD previous pair objString occurences!!!! (added in pathline string)
					if (i>0) {
							
						for (int o=0; o<Enriched_DTI_BLKG.SEMANTIC_TYPES; o++)
							subjectSemOccurences[o]+=objectSemOccurences[o];
						for (int o=0; o<Enriched_DTI_BLKG.SEMANTIC_TYPES; o++)
							objectSemOccurences[o]=0;
					}
					
					for (int j=0; j<Enriched_DTI_BLKG.NON_NORMALIZED_RELATIONS; j++) {
	    	        	 
	    	        	String relTypeLine=nodePair_relationsOcc[j]; 
						if ((relTypeLine!=null) &&(!relTypeLine.equals(""))) {
							String [] occurences = relTypeLine.split("-");
							String subjString = occurences[0];
							String relString = occurences[1];
								
							if (!subjString.equals("")) {
								String [] subjStrings = subjString.split("@");
								for (String subjectType: subjStrings) {
									String []subjArray = subjectType.split("x");
									if (subjArray.length>1) {
										int semTypeIndex = nodeSemTypesList.indexOf(subjArray[1]);
										subjectSemOccurences[semTypeIndex]+=Integer.parseInt(subjArray[0]);
									}	
								}
							}	
							String [] relArray = relString.split("x");
							if (relArray[0].equals(""))
								System.out.println("PROBLEM in "+relTypeLine+" pair: "+pair+ " ");
							int relOcc = Integer.parseInt(relArray[0]);
							String relAbnormalType = relArray[1];
							String relType = terms.normalizedNames.get(relAbnormalType);
							int index = terms.relationTypes.indexOf(relType);
							relationSemOccurences[index] += relOcc;
							
						//	if ((i!=l) && (occurences.length<3)) {
						//		System.out.println("OOPS! PROBLEM WITH PAIR: "+pair+" in line: "+line);
						//		System.out.println("&&&&& no object in relTypeLine="+relTypeLine);
						//	}
							if (i<l) {
								
								if (!subjString.equals("")) {
									String objString = occurences[2];
										
									String [] objStrings = objString.split("@");
									for (String objectType: objStrings) {
										String []objArray = objectType.split("x");
										if (objArray.length>1) {
											int semTypeIndex = nodeSemTypesList.indexOf(objArray[1]);
											objectSemOccurences[semTypeIndex]+=Integer.parseInt(objArray[0]);
										}
									}
								}
							}
						}	
					}
				}
				
				for (int j=0; j<Enriched_DTI_BLKG.SEMANTIC_TYPES; j++) 
					pairline[i*offset+j]+=subjectSemOccurences[j];
				for (int j=0; j<Enriched_DTI_BLKG.RELATIONS; j++)
					pairline[i*offset+Enriched_DTI_BLKG.SEMANTIC_TYPES+j]+=relationSemOccurences[j];

			}
			
			
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
		
		
		//feature line for this pair is ready....now print in output csv
		FileWriter fw = new FileWriter(outFile, true);
		BufferedWriter bw = new BufferedWriter(fw);
		PrintWriter out = new PrintWriter(bw);
		
		out.print(drug+"_"+target+",");
		for (int i=0; i<Enriched_DTI_BLKG.LENGTH; i++)
			for (int j=0; j<offset; j++) 
				out.print(pairline[i*offset+j]+",");
		out.println(groundtruth);

		out.close();
		bw.close();
		fw.close();
	}
}
