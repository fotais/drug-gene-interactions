import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
//import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.stream.*;

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

public class SaveTopNPathsForPosPairs {

	final static int N=100;
	
	public static void main (String []args) {

	   	String drugPairsFolder = args[0];
    	String neo4jFolder = args[1];
       	final int length=3;
    	int pathFilesAlreadyexisted=0;
    	
        //File databaseDirectory = new File("C:\\FOT\\neo4j-community-3.5.23\\data\\databases\\graph.db");
    	File databaseDirectory = new File(neo4jFolder);
    	
		GraphDatabaseService graphDb = new GraphDatabaseFactory().newEmbeddedDatabase( databaseDirectory );
		
        registerShutdownHook( graphDb );
        
        Node firstNode;
        Node secondNode;
        HashMap<String,Integer> paths_freq = new HashMap<String,Integer>(); 
        
        boolean includesLoop=false;
        
        try ( Transaction tx = graphDb.beginTx() )
        {
        	
        	try {
        		
 	           
        		String drugPair;
        		FileReader filerdr = new FileReader(drugPairsFolder+"ProcessNeo4j/PositivePairs/drug-pairs.csv");
        		BufferedReader in = new BufferedReader(filerdr);
	            int n=0;

        		while(( drugPair = in.readLine() ) != null   ) {
        			n++;
        	        String[] drugs = drugPair.split(",");
         	
        	            	        
		          	Label l1 = Label.label("Entity"); 
		          	Label l2 = Label.label("Entity");
		            firstNode = graphDb.findNode(l1, "id", drugs[0]);
		            secondNode = graphDb.findNode(l2, "id", drugs[1]);
		            if ((firstNode==null) || (secondNode==null))
		            	continue;
		            
 		            PathFinder<Path> finder = GraphAlgoFactory.allSimplePaths(PathExpanders.allTypesAndDirections(), length);
		            Iterable<Path> paths = finder.findAllPaths(firstNode, secondNode);
		            
		            System.out.println("Examining all paths of pair: +"+drugs[0]+"_"+drugs[1]);
		            for (Path p : paths){
		         
		            	
		            	String rchain = "";
			            Node currentNode = firstNode;            	
		            	Iterable<Relationship> relations = p.relationships();
		            	int rNo=0;
		            	for (Relationship r : relations){
		            		if (currentNode.equals(r.getOtherNode(currentNode))) {
		            			includesLoop=true;
		            			System.out.println("FOUND LOOP!");
		            			break;
		            		}	
		            		String rtype = r.getType().name();
		            		String [] resources= (String []) r.getProperty("resource");
	            			if (((rtype.equals("INTERACTS_WITH")) || (rtype.equals("INTERACTS_WITH__INFER__")) || (rtype.equals("INTERACTS_WITH__SPEC__"))) && (!Arrays.asList(resources).contains("DRUGBANK"))) {
	            				List<String> subjectTypes = Arrays.asList((String []) r.getProperty("subject_sem_type"));
	            				List<String> objectTypes = Arrays.asList((String []) r.getProperty("object_sem_type"));
	            				if ((((subjectTypes.contains("orch")) || (subjectTypes.contains("phsu"))) && ((objectTypes.contains("gngm")))) || ((((objectTypes.contains("orch")) || (objectTypes.contains("phsu"))) && ((subjectTypes.contains("gngm")))))){
	            					rtype="LITERATURE_DTI";
	            				}	
	            			}	
		            		if(rNo++!=0)
		            			rchain =rchain+",";
		            		rchain =rchain+rtype;
		            		currentNode=r.getOtherNode(currentNode);
		            	}
		            	if (includesLoop) 
		            		continue;	
		            	
		            	//JUST FOR TESTING
		            	//if (p.length()==1)
		            		//System.out.println("rchain added->"+rchain);
		            	
		            	Integer freq;
		            	if ( (freq= paths_freq.get(rchain))==null)
		            		paths_freq.put(rchain, new Integer(1));
		            	else
		            		paths_freq.put(rchain,freq.intValue()+1); 
		            }
		            tx.success();
		            
		            
		            /*
	            	if (++n>100000) {
	            		System.out.println("Examined "+n+" paths");
	            		HashMapSorting.sortAndSave(N, writer, paths_freq);
	            		//empty due to mem limitations
	            		paths_freq = new HashMap<String,Integer>();
	            	*/
		            
	            }
		        in.close();
				filerdr.close();

        		
        		//FOT ADDITION - minus neg pairs popular paths - checking n random neg pairs
        		//FileReader filerdr2 = new FileReader();
        		//BufferedReader in2 = new BufferedReader(filerdr2);
				java.nio.file.Path path = Paths.get(drugPairsFolder+"ProcessNeo4j/NegativePairs/drug-pairs.csv");
        		Random random = new Random();
	 			while(--n>0 ) {
	 				Stream<String> lines = Files.lines(path);
	        		String drugPair2 = lines.skip(random.nextInt(79000)).findFirst().get();
	 				//String  drugPair2 = in2.readLine() ;
        	   		String[] drugs = drugPair2.split(",");
              	  	Label l1 = Label.label("Entity"); 
		          	Label l2 = Label.label("Entity");
		            firstNode = graphDb.findNode(l1, "id", drugs[0]);
		            secondNode = graphDb.findNode(l2, "id", drugs[1]);
		            if ((firstNode==null) || (secondNode==null))
		            	continue;
		            PathFinder<Path> finder = GraphAlgoFactory.allSimplePaths(PathExpanders.allTypesAndDirections(), length);
		            Iterable<Path> paths = finder.findAllPaths(firstNode, secondNode);
		            System.out.println("Examining all paths of negative pair: +"+drugs[0]+"_"+drugs[1]);
		            for (Path p : paths){
		            	String rchain = "";
			            Node currentNode = firstNode;            	
		            	Iterable<Relationship> relations = p.relationships();
		            	int rNo=0;
		            	for (Relationship r : relations){
		            		if (currentNode.equals(r.getOtherNode(currentNode))) {
		            			includesLoop=true;
		            			System.out.println("FOUND LOOP!");
		            			break;
		            		}
		            		String rtype = r.getType().name();
		            		String [] resources= (String []) r.getProperty("resource");
		            		if (((rtype.equals("INTERACTS_WITH")) || (rtype.equals("INTERACTS_WITH__INFER__")) || (rtype.equals("INTERACTS_WITH__SPEC__"))) && (!Arrays.asList(resources).contains("DRUGBANK"))) {
	            				List<String> subjectTypes = Arrays.asList((String []) r.getProperty("subject_sem_type"));
	            				List<String> objectTypes = Arrays.asList((String []) r.getProperty("object_sem_type"));
	            				if ((((subjectTypes.contains("orch")) || (subjectTypes.contains("phsu"))) && ((objectTypes.contains("gngm")))) || ((((objectTypes.contains("orch")) || (objectTypes.contains("phsu"))) && ((subjectTypes.contains("gngm")))))){
	            					rtype="LITERATURE_DTI";
	            				}	
	            			}	
		            		if(rNo++!=0)
		            			rchain =rchain+",";
		            		rchain =rchain+rtype;
		            		currentNode=r.getOtherNode(currentNode);
		            	}
		            	if (includesLoop) 
		            		continue;	
		            	Integer freq;
		            	if ( (freq= paths_freq.get(rchain))!=null)
		            		paths_freq.put(rchain,freq.intValue()-1);
		            	
		            	lines.close();
		            }	
        		}
	 			tx.success();
	 			//FOT ADDITION END
        			

	 			//Write top N paths in an output file
    	        File f = new File(drugPairsFolder+"ProcessNeo4j/PositivePairs/Top"+N+"Paths.txt");
    	        PrintWriter writer= new PrintWriter(f, "UTF-8");
        		HashMapSorting.sortAndSave(N, writer, paths_freq);
	            
                writer.close();
	            
	         }catch(Exception e) {
	            	e.printStackTrace();
	         }
	    }
	
	    System.out.println("ALL PATHS RETRIEVED");
	    System.out.println("Skipped "+pathFilesAlreadyexisted+ " drug pairs that their paths were already retrieved in a file...");
        
        graphDb.shutdown();
    }
	
	private static void registerShutdownHook( final GraphDatabaseService graphDb )
    {
        // Registers a shutdown hook for the Neo4j instance so that it
        // shuts down nicely when the VM exits (even if you "Ctrl-C" the
        // running application).
        Runtime.getRuntime().addShutdownHook( new Thread()
        {
            @Override
            public void run()
            {
                graphDb.shutdown();
            }
        } );
    }  
}
