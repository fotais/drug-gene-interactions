/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
//import java.util.Set;

import org.neo4j.graphalgo.GraphAlgoFactory;
import org.neo4j.graphalgo.PathFinder;
import org.neo4j.graphdb.GraphDatabaseService;
import org.neo4j.graphdb.Label;
import org.neo4j.graphdb.Node;
import org.neo4j.graphdb.Path;
import org.neo4j.graphdb.PathExpanders;
import org.neo4j.graphdb.Relationship;
import org.neo4j.graphdb.Transaction;
import org.neo4j.graphdb.factory.GraphDatabaseFactory;


 
/**
 *
 * @author fot
 */
public class Enriched_DTI_BLKG {

	final static int RELATIONS = 36; 
	final static int NON_NORMALIZED_RELATIONS=70;
	final static int SEMANTIC_TYPES = 127;
	final static int LENGTH=3;
	final static int FEATURES = LENGTH*RELATIONS+LENGTH*SEMANTIC_TYPES;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args)  {
    	
    	String semaTyPfolder = args[0];
    	String neo4jFolder = args[1];
    	String groundtruth = args[2];
    	String drugPairsFolder;
    	if (groundtruth.equals("1"))
    		drugPairsFolder= semaTyPfolder+"ProcessNeo4j/PositivePairs/";
    	else
    		drugPairsFolder= semaTyPfolder+"ProcessNeo4j/NegativePairs/";
    	SemanticTypesRetriever types= new SemanticTypesRetriever();
    	String [] nodeSemTypes = types.getNodeSemTypesArray(semaTyPfolder);
    	List<String> nodeSemTypesList = types.getNodeSemTypes(semaTyPfolder);
    	Terms terms = new Terms();
    	String [] relSemTypes = types.getRelSemTypes(terms.relationTypes);

    	HashMap<String,String[]> nodePairs_relationsOcc = new HashMap<String,String[]>();
    	//int hashcounter1 = 0;
    			
    	int pathFilesAlreadyexisted=0;
    	
        //File databaseDirectory = new File("C:\\FOT\\neo4j-community-3.5.23\\data\\databases\\graph.db");
    	File databaseDirectory = new File(neo4jFolder);
    	
        GraphDatabaseService graphDb = new GraphDatabaseFactory().newEmbeddedDatabase( databaseDirectory );
        registerShutdownHook( graphDb );
        
        Node firstNode;
        Node secondNode;
        
        
        try ( Transaction tx = graphDb.beginTx() )
        {
        	
        	try {
        		String featureFile = semaTyPfolder+"FeatureExtraction/features.csv";
        		
        		//if positive samples are retrieved, create a new feature file
        		if (groundtruth.equals("1")){
	        	    PrintWriter writer2= new PrintWriter(featureFile, "UTF-8");
		            //write header with feature names
		            writer2.print("Drug_Target,");
		            for (int j=1; j<=LENGTH; j++) {
	        	        for (int i=0; i<SEMANTIC_TYPES; i++)
	        	        	writer2.print("nod"+j+"_"+nodeSemTypes[i]+",");
	        	        for (int i=0; i<RELATIONS; i++)
	        	        	writer2.print("rel"+j+"_"+relSemTypes[i]+","); 
		            }
		            writer2.println("GROUNDTRUTH");
		            writer2.close();
        		}
        		
        		
        		String drugPair;
        		FileReader filerdr = new FileReader(drugPairsFolder+"drug-pairs.csv");
        		BufferedReader in = new BufferedReader(filerdr);
        		while(( drugPair = in.readLine() ) != null   ) {
        	        String[] drugs = drugPair.split(",");
         	
        	        //FOT to remove - just for testing
        	        //if (++pairs==5)
        	        	//break;
        	        
        	        //skip this drug pair, if the file with the paths already exists!
        	        String fname = drugPairsFolder+"ExtractedPath_Files/"+drugs[0]+"_"+drugs[1]+"_paths_"+LENGTH+".csv";
        	        File f = new File(fname);
        	        if (f.exists()) {
        	        	pathFilesAlreadyexisted++;
        	        	continue;
        	        }
        	        
		          	Label l1 = Label.label("Entity"); 
		          	Label l2 = Label.label("Entity");
		            firstNode = graphDb.findNode(l1, "id", drugs[0]);
		            secondNode = graphDb.findNode(l2, "id", drugs[1]);
		            if ((firstNode==null) || (secondNode==null))
		            	continue;
		            
		            //DEBUG
		            System.out.println("looking for paths to save in file: "+drugs[0]+"_"+drugs[1]+"_paths_"+LENGTH+".csv");
		            
		            PathFinder<Path> finder = GraphAlgoFactory.allSimplePaths(PathExpanders.allTypesAndDirections(), LENGTH);
		            Iterable<Path> paths = finder.findAllPaths(firstNode, secondNode);

		            //DEBUG
		            int pathC=0;
		            int pathsE=0;
		            int dpaths=0;
		            int loops=0;
		            
		            //PrintWriter writer= new PrintWriter(fname, "UTF-8");
	            	java.util.Collection<String> unique_cui_paths = new  ArrayList<String>();
	            	java.util.HashSet<String> node_cui_paths = new java.util.HashSet<String>(10000000);

	            	//DEBUG
		            //System.out.println("DEBUG: paths Iterable<path> paths class: "+paths.getClass());

		            for (Path p : paths){
		            	pathsE++;
		            	
		                boolean includesLoop=false;
		                String pathLine="";  
		            	
		            	int n=0;
		            	//first save node cuis
		            	String [] cuis = new String[LENGTH+1];
		                for (Node node:p.nodes()) {
		                	String cui = (String)node.getProperty("id");
		                	cuis[n]=cui;
		                	//System.out.println("next Cui="+cui);
		                	if (n!=0) 
		                		pathLine+=cui+",";
		                	//System.out.println("pathLine="+pathLine);
		                	n++;
		                }	
		                
		                includesLoop=checkDuplicateUsingAdd(cuis);
		            	if (includesLoop){
		            		loops++;
		                	System.out.print("DEBUG: path including LOOP: ");
		                	for (String c: cuis)
		                		System.out.print(c+"-");
		                	System.out.println("");
		                	System.out.print("Path length="+p.length());
		            		continue;
		            	}
		            	pathLine=pathLine.substring(0,pathLine.length()-1);
		            	//System.out.println("pathLine="+pathLine);
		            	
		            	n=0;
		            	String [] rels = new String[4];
		            	for (Relationship r: p.relationships()) {
		            		rels[n++]=r.getType().name();
		            	}
		            	String uniquePath = ""+Arrays.toString(cuis)+Arrays.toString(rels);
		            		
		            	//now check if the exact path has already been saved and if not, save in node paths
		            	if (unique_cui_paths.contains(uniquePath)) {
		            		dpaths++;
		            		continue;
		            	}


		            	//for valid paths shorter than LENGTH, complete with target nodes
	                	if (p.length()<LENGTH){
	                		//Just for debug
	                		if (p.length()==LENGTH-2) {
	                			pathLine+=","+cuis[n]+","+cuis[n];
	                			System.out.println("FOUND Direct relation=" +pathLine);
	                		}	
	                		else if (p.length()==LENGTH-1)
	                			pathLine+=","+cuis[n];
                       	}
		            	unique_cui_paths.add(uniquePath);
		            	node_cui_paths.add(pathLine);

	                	int step=0;
	                	Node currentNode = firstNode;            	
		            	Iterable<Relationship> relations = p.relationships();
		            	for (Relationship r : relations){
		            		step++;
		            		
	            			//check if cui pair AND relation type is already saved in the general hashmap
		            		String cui_pair = cuis[step-1]+cuis[step];
		            		String [] relationTypes;
		            		
		            		String nonNormrelType = r.getType().name();
		            		int relIndex1 = terms.nonNormrelationTypes.indexOf(nonNormrelType);
		            		if (relIndex1==-1) {
		            			System.out.println("Found erroneous Non-normalized relation:" + nonNormrelType);
		            			break;
		            		}
		            			
		            		if ((relationTypes=(String [])nodePairs_relationsOcc.get(cui_pair))!=null) { 
		            			if (relationTypes[relIndex1]!=null) {
		            				currentNode=r.getOtherNode(currentNode);
		            				continue;
		            			}	
		            		}
		            		else
		            			relationTypes= new String[NON_NORMALIZED_RELATIONS]; //RELATIONS];
		            		//if not we must save the occurence string in the String array...
		            		String occurenceString = "";
		            		
		            		String [] resources= (String []) r.getProperty("resource");
	            			int relOccurencies =resources.length;
	            			
	            			//SPECIAL CASE: If relation is Interacts_with, we decided to scale from 0-100 (drugbank rel=100)
	            			if ((nonNormrelType.equals("INTERACTS_WITH")) || (nonNormrelType.equals("INTERACTS_WITH__INFER__")) || (nonNormrelType.equals("INTERACTS_WITH__SPEC__"))) {
            					
	            				// Case 1: DDI from drugbank=> score=100
	            				if (Arrays.asList(resources).contains("DRUGBANK"))
	            					relOccurencies=100;
	            				// Case 2: interaction from literature=> scale score 1-100 (depends if its a DDI or other type interaction) 
	            				else{
		            				List<String> subjectTypes = Arrays.asList((String []) r.getProperty("subject_sem_type"));
		            				List<String> objectTypes = Arrays.asList((String []) r.getProperty("object_sem_type"));

	            					//Case of both nodes being drugs (literature-identified DDIs)
		            				if (((subjectTypes.contains("orch")) || (subjectTypes.contains("phsu"))) && ((objectTypes.contains("orch")) || (objectTypes.contains("phsu"))))
		            					relOccurencies=(int)Math.round((relOccurencies*100)/15);  //15 is the max possible occurrences for DDIs from different articles
	            					//Interesting case of drug-target interaction (literature-identified DTIs)
		            				else if ((((subjectTypes.contains("orch")) || (subjectTypes.contains("phsu"))) && ((objectTypes.contains("gngm")))) || ((((objectTypes.contains("orch")) || (objectTypes.contains("phsu"))) && ((subjectTypes.contains("gngm")))))){
		            					nonNormrelType="LITERATURE_DTI";
		            					relIndex1 = terms.nonNormrelationTypes.indexOf(nonNormrelType);
		            				}	
		            				else {
		            					relOccurencies=(int)Math.round((relOccurencies*100)/200);  //200 is the max possible occurrences from Articles
		            				}	
	            				}
		            			//just debug printing....
	            				//if (pathsE<300)
	            					//System.out.println("Found "+nonNormrelType+". relOccurences scaled="+relOccurencies);
	            			}
	            					
	            			//set subject/object semantic types based on direction
	            			String [] subjectSemTypes = {};
	            			String [] objectSemTypes = {};
	          
	            			if (!((nonNormrelType.equals("HAS_MESH") || (nonNormrelType.equals("MENTIONED_IN")) || (nonNormrelType.equals("IS_A"))))) {
	            				
	            				//for negations of relation count as -1 in rel occurence!
	            				int neg=0;
	            				if (!r.hasProperty("negation"))
	            					;//System.out.println("Problematic relation with missing properties found: "+r.getType().toString());
	            				else {
	            					String [] negations= (String []) r.getProperty("negation");
	            				
	            					for (String negation: negations)
		            					if (negation.equals("negation"))
		            						neg++;
		            				relOccurencies-=neg;
		            				if (relOccurencies<=0) {
		            					//System.out.println("DEBUG: In the hop of CUIs: "+cui_pair+" rel Type="+nonNormrelType+" has polarity="+relOccurencies);
		            					currentNode=r.getOtherNode(currentNode);
		            					continue;
		            				}	
	            				}
	            				
	            				if (r.hasProperty("subject_sem_type")) {
			            			if (currentNode.equals(r.getStartNode())) {
		           						subjectSemTypes = (String []) r.getProperty("subject_sem_type");
		           						objectSemTypes = (String []) r.getProperty("object_sem_type");
			            			}else {
			            				//System.out.println("Relation "+r.getType()+" from "+currentNode.getId()+" has left direction...");
		           						objectSemTypes= (String []) r.getProperty("subject_sem_type");
			            				subjectSemTypes = (String []) r.getProperty("object_sem_type");	            				
			            			}
	            				}	
	            			}	
	            			currentNode=r.getOtherNode(currentNode);

	            			//initiate all types with 0 
	            			HashMap<String,Integer> subjectSemTypeOccurences = new HashMap<String,Integer>(LENGTH); 
	            			for (String type:nodeSemTypes)    
	            				subjectSemTypeOccurences.put(type, 0);
	            			for (String typeFound:subjectSemTypes) {
	            				int addOccurence = (subjectSemTypeOccurences.get(typeFound)).intValue();
	            				subjectSemTypeOccurences.put(typeFound,++addOccurence);
	            			}
	            			
							//initiate all types with 0
	            			HashMap<String,Integer> objectSemTypeOccurences = new HashMap<String,Integer>(LENGTH); 
	            			for (String type:nodeSemTypes)   
	            				objectSemTypeOccurences.put(type, 0);
	            			for (String typeFound:objectSemTypes) {
	            				int addOccurence = (objectSemTypeOccurences.get(typeFound)).intValue();
		            			objectSemTypeOccurences.put(typeFound, ++addOccurence);
		            			//System.out.println("Add an object occurence "+addOccurence+" for "+typeFound);
	            			}
            				int occurences;
            				
            				
	            			for (String type:nodeSemTypes)  
	            				if ((occurences= (subjectSemTypeOccurences.get(type)).intValue())>0)
	            					occurenceString+=occurences+"x"+type+"@";
	            			if (occurenceString.equals(""))
	            				occurenceString+="@";
	            			occurenceString=occurenceString.substring(0, occurenceString.length()-1)+"-";
	            			occurenceString+=relOccurencies+"x"+nonNormrelType+"-";
	            			for (String type:nodeSemTypes)  
			            		if ((occurences= (objectSemTypeOccurences.get(type)).intValue())>0)
			            				occurenceString+=occurences+"x"+type+"@";

	            			if (!(occurenceString.charAt(occurenceString.length()-1)=='@'))
	            				occurenceString+="@";
	            			occurenceString=occurenceString.substring(0, occurenceString.length()-1);
		            			
	            			relationTypes[relIndex1]=occurenceString;
	            			nodePairs_relationsOcc.put(cui_pair, relationTypes);
	            			
		            	}
		            	pathC++;

		            	if ((pathC%1000000)==0) {
		            			System.out.println("---------------------------------------");
			            		//System.out.println("DEBUG: paths examined reached "+pathsE +".");
			            		System.out.println("DEBUG: interesting paths reached "+pathC +". now saving in file...");
			            		//System.out.println("DEBUG: loops found="+loops+" and duplicate paths retrieved from API="+dpaths);
			            		System.out.println("DEBUG: nodePairs_relationOcc hashmap size reached: "+nodePairs_relationsOcc.size());
			            		PrintWriter writer= new