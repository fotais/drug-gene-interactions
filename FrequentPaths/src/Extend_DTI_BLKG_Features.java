import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;

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

public class Extend_DTI_BLKG_Features {

	public static void main (String [] args) {
		
		String featuresPath = args[0]+"FeatureExtraction/";
		String neo4jFolder = args[1];
		final int length=3;
		final int N = SaveTopNPathsForPosPairs.N;
		
		//FIRST save frequent paths list in a ArrayList object
		ArrayList<String> frPaths = new ArrayList<String> ();
		try {
			FileReader filerdr1 = new FileReader(args[0]+"ProcessNeo4j/PositivePairs/Top"+N+"Paths.txt");
			BufferedReader read = new BufferedReader(filerdr1);
			int n=0;
			String freqPathLine;
	        while ((freqPathLine = read.readLine())!=null)
				frPaths.add(n++, freqPathLine);
			read.close();
			filerdr1.close();
		}catch (Exception e) {
			e.printStackTrace();
		}
		
		//now open Neo4j to retrieve all paths...
        File databaseDirectory = new File(neo4jFolder);
    	GraphDatabaseService graphDb = new GraphDatabaseFactory().newEmbeddedDatabase( databaseDirectory );
		registerShutdownHook( graphDb );
        
        Node firstNode;
        Node secondNode;
        int[] paths_freq ; 
        
        try ( Transaction tx = graphDb.beginTx() )
        {
        	
        	try {
    			FileReader filerdr = new FileReader(featuresPath+"DTI-enriched_sematyp_features.csv");
    			BufferedReader in = new BufferedReader(filerdr);

  	           //Write previous + frequent path features in a new file
     	        File featuresExtended = new File(featuresPath+"DTI-enriched_sematyp_featuresExtended.csv");
     	        PrintWriter writer= new PrintWriter(featuresExtended, "UTF-8");

     	        //Add feature labels in the extended features file
	            String labels = in.readLine();
	            for (int i=0; i< N; i++)
	            	labels=labels+",PATH"+i;
	            writer.println(labels);
	            
	            int pnum=0;
	            String featureLine;
        		while(( featureLine = in.readLine() ) != null   ) {
        			
        			//add frequent relations/sem type features
        			writer.print(featureLine);
        			
        			pnum++;
        			//and now time to calculate frequent path features...
        			
        			//initialise them with 0s
        			paths_freq = new int[N];
        			for (int i=0; i<N; i++)
        				paths_freq[i]=0;
        			
        			//and then count in paths
        	        String[] items = featureLine.split(",");
        	        String[] drugs = items[0].split("_");
         	
		          	Label l1 = Label.label("Entity"); 
		          	Label l2 = Label.label("Entity");
		            firstNode = graphDb.findNode(l1, "id", drugs[0]);
		            secondNode = graphDb.findNode(l2, "id", drugs[1]);
		            if ((firstNode==null) || (secondNode==null))
		            	continue;
		            
 		            PathFinder<Path> finder = GraphAlgoFactory.allSimplePaths(PathExpanders.allTypesAndDirections(),length);
		            Iterable<Path> paths = finder.findAllPaths(firstNode, secondNode);
		            
		            //System.out.println("Examining all paths of pair: +"+drugs[0]+"_"+drugs[1]);
		            for (Path p : paths){
		            	
		            	boolean includesLoop=false;
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
		            		if(rNo++!=0)
		            			rchain =rchain+",";
		            		//else if(firstNode.equals(r.getOtherNode(currentNode))) {
		            			//includesLoop=true;
		            			//System.out.println("FOUND LOOP!");
		            			//break;
		            		//}
		            		rchain =rchain+r.getType().name();
		            		currentNode=r.getOtherNode(currentNode);
		            	}
		            	if (includesLoop) 
		            		continue;	

		            	if (frPaths.contains(rchain)) {
		            		//find index
		            		int i = frPaths.indexOf(rchain);
		            		if ((i<0) ||(i>99))
				            	System.out.println("PROBLEM! found path "+rchain+" in freq paths list index "+i+" which is out of bounds!!!");
		            		paths_freq[i]+=1;
		            	}
		            }
		            tx.success();
		            if ((pnum%500)==0)
		            	System.out.println("Extended features for "+pnum+" pairs");
		            //System.out.println("Finished pair: +"+drugs[0]+"_"+drugs[1]);
		            for (int i=0; i<N; i++)
		            	writer.print(","+paths_freq[i]);
		            writer.println();
	            	
		        }
	            in.close();
	            filerdr.close();
        		writer.close();
	            
	         }catch(Exception e) {
	            	e.printStackTrace();
	         }
	    }
	
	    System.out.println("Finished feature extension...");
        
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
