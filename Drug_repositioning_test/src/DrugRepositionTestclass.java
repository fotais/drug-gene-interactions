import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.LinkedList;


public class DrugRepositionTestclass {
	
	static int t=0;
	static int d=0;

	public static void main(String [] args) {

		try {

			String ddiPath = args[0];

			String method = "DTI-BLKG";
			
			String toppred = ddiPath+"TOP PREDICTIONS_"+method+".csv";
			int f=0;
			int t=0;
			
			PrintWriter conf= new PrintWriter(ddiPath+"dtis-validation_"+method+".csv", "UTF-8");
			conf.println("Pred_Pair,Conf_Score,SNAP,OpenTargets,bindingDB,STITCH,CTD,DrugCentral,Any of these");
			BufferedReader brTop = new BufferedReader(new FileReader(toppred));
			//BufferedReader brdrugs = new BufferedReader(new FileReader(drugmap1));
			//BufferedReader brtargets = new BufferedReader(new FileReader(targetmap1));
			String line;
			while ((line = brTop.readLine()) != null ){
				String[] values = line.split(",");
				double pred = new Double(values[1]);
				String pair = values[0];
				String [] drugs = pair.split("_");
				String drug=drugs[0];
				String target=drugs[1];
				if ((pred>0.5) && (t<50)){
					
					String dbid = getUMLSMapping(drug, ddiPath);
					String drName = getUMLSNameMapping(drug, ddiPath);
					String cas_num = getUMLS_CASMapping(drug, ddiPath);
					String bindid = getBindingDBmapping(dbid, ddiPath);
					String uniid = getUniprotMapping(target, ddiPath);
					String uniname = getUniprotMapping2(target, ddiPath);
					String siderid = getSIDERMapping(dbid, ddiPath);
					String gene = mapTargetCUItoName(target, ddiPath);
					String ensembl = mapTargetCUItoEnsembl(target, ddiPath);
					String stringId = mapTargetEnsembltoStringId(gene, "/media/fot/OS/Downloads/9606.protein.info.v11.5.txt");
					String chembl = mapCUItoChemBL(drug, ddiPath);
					if (siderid.equals("null"))
						siderid = getSIDERMapping2(dbid, ddiPath);
					String stitchId = siderid.replaceAll("CID0", "CIDs");
					stitchId = stitchId.replaceAll("CID1", "CIDm");
					String disgid = getDisGenetMapping(uniid, ddiPath);
					System.out.println("Cross-checking pred "+stitchId+"-"+stringId+" score="+pred);

					if (existsInPosGroundtruth(dbid,gene, ddiPath+"Full_Aggregated_DTIs_GroundTruth_Pos.csv")) {
						System.out.println("HOUSTON WE HAVE A PROBLEM. In Pos Groundtruth I have found "+dbid+","+gene);
						continue;
					}
						
					
					t++;
					boolean found=false;
					conf.print(pair+","+pred+",");
					
					
					if (foundInChGMiner(dbid,uniid, ddiPath+"SNAP/ChG-Miner_miner-chem-gene.tsv")) {
						System.out.println("found "+dbid+","+uniid);
						found=true;
						conf.print("YES,");
					}	
					else if (foundInInterDecagon_targets(siderid,disgid, ddiPath+"SNAP/ChG-InterDecagon_targets.csv")) {
						System.out.println("found "+siderid+","+disgid);
						found=true;
						conf.print("YES,");
					}
					else
						conf.print("NO,");
					
					if (foundInOpentargets(gene,chembl, ddiPath+"OpenTargets.ORG/tractability_buckets-2021-01-12.tsv")) {
						System.out.println("found in OpenTargets "+gene+","+chembl);
						found=true;
						conf.print("YES,");
					}
					else
						conf.print("NO,");
					if (foundbindingDB(uniid,bindid, ddiPath+"bindingdb/purchase_target_10000.tsv")) {
						System.out.println("found in bindingDB "+uniid+","+bindid);
						found=true;
						conf.print("YES,");
					}
					else
						conf.print("NO,");
					if (foundInStitch(stringId,stitchId, "/media/fot/OS/Downloads/stitch_9606.protein_chemical.links.v5.0.tsv")) {
						System.out.println("found in STITCH "+stitchId+","+stringId);
						found=true;
						conf.print("YES,");
					}
					else
						conf.print("NO,");
					
					if (foundInCTD(gene,drName,cas_num, "/media/fot/OS/Downloads/CTD_chem_gene_ixns.tsv")) {
						System.out.println("found in CTD "+stitchId+","+stringId);
						found=true;
						conf.print("YES,");
					}
					else
						conf.print("NO,");
					
					if (foundInDrugCentral(uniname,drName, ddiPath+"DrugCentral/DrugCentral_drug.target.interaction.tsv")) {
						System.out.println("found in DrugCentral "+uniname+","+dbid);
						found=true;
						conf.print("YES,");
					}
					else
						conf.print("NO,");
					
					if (found) {
						f++;
						conf.println("YES");
					}
					else
						conf.println("NO");
				}
			}
			System.out.println("Confirmed "+f+" out of "+t);
			brTop.close();
			conf.close();
			//brtargets.close();
			//brdrugs.close();
					
			
			//debug
			System.out.println("Finished ...");

		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
	}
	
	

	
	static String getUMLSMapping(String cui, String ddiPath) {
		
   		String id="null";
		String line;
		String drugMappings = ddiPath+"drug-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
			        String[] values = line.split("\t");
			        id = values[0];
			        break;
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		if (id.equals("null"))
				++t;//System.out.println((++t)+" null DB id for CUI: "+cui);
		return id;

	}


	static String getUMLSNameMapping(String cui, String ddiPath) {
		
   		String id="null";
		String line;
		String drugMappings = ddiPath+"drug-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
			        String[] values = line.split("\t");
			        id = values[1];
			        break;
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		if (id.equals("null"))
				++t;//System.out.println((++t)+" null DB id for CUI: "+cui);
		return id;

	}
	
	static String getUMLS_CASMapping(String cui, String ddiPath) {
		
   		String id="null";
		String line;
		String drugMappings = ddiPath+"drug-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
			        String[] values = line.split("\t");
			        id = values[4];
			        break;
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		if (id.equals("null"))
				++t;//System.out.println((++t)+" null DB id for CUI: "+cui);
		return id;

	}
	
	static String getBindingDBmapping (String cui, String ddiPath) {
		
   		String id="null";
		String line;
		String drugMappings = ddiPath+"drug-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
			        String[] values = line.split("\t");
			        id = values[10];
			        break;
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		if (id.equals("null"))
				++t;//System.out.println((++t)+" null DB id for CUI: "+cui);
		return id;

	}

	static String getUniprotMapping(String cui, String ddiPath) {
		
   		String id="null";
		String line;
		String drugMappings = ddiPath+"target-mappings_upd.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
			        String[] values = line.split("\t");
			        id = values[2];
			        break;
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		if (id.equals("null"))
				++t;//System.out.println((++t)+" null DB id for CUI: "+cui);
		return id;

	}

	static String getUniprotMapping2(String cui, String ddiPath) {
		
   		String id="null";
		String line;
		String drugMappings = ddiPath+"target-mappings_upd.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
			        String[] values = line.split("\t");
			        id = values[1];
			        break;
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		if (id.equals("null"))
				++t;//System.out.println((++t)+" null DB id for CUI: "+cui);
		return id;

	}
	
	static String getSIDERMapping(String db, String ddiPath) {
		
   		String id="null";
		String line;
		String drugMappings = ddiPath+"DrugBank-Sider_mapping.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(db)) {
			        String[] values = line.split("\t");
			        id = values[3];
			        break;
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		if (id.equals("null"))
				++t;//System.out.println((++t)+" null DB id for CUI: "+cui);
		return id;

	}


	static String getSIDERMapping2(String db, String ddiPath) {
		
   		String name="0000000";
		String line;
		String drugMappings1 = ddiPath+"drug-mappings_latest.tsv";
		try {
			BufferedReader br1 = new BufferedReader(new FileReader(drugMappings1));
	   		
			while ((line = br1.readLine()) != null ){
				if(line.contains(db)) {
			        String[] values = line.split("\t");
			        name = values[1];
			        break;
				}    
			}
			br1.close();
		}catch(Exception e) {
			e.printStackTrace();
		}

		String id="------";
		String drugMappings2 = ddiPath+"drug_SIDER_names.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings2));
	   		
			while ((line = br.readLine()) != null ){
				String[] values = line.split("\t");
				if (values[1].equalsIgnoreCase(name)) {
			        id = values[0];
			        break;
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		return id;

	}

	static String getDisGenetMapping(String unip, String ddiPath) {
		
   		String id="null";
		String line;
		String drugMappings = ddiPath+"DisGenet_uniprot_mapping.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(unip)) {
			        String[] values = line.split("\t");
			        id = values[1];
			        break;
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		if (id.equals("null"))
				++t;//System.out.println((++t)+" null DB id for CUI: "+cui);
		return id;

	}

	static String mapTargetCUItoName(String cui, String ddiPath) {
		
   		String name="null";
		String line;
		String drugMappings = ddiPath+"target-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
					
			        String[] values = line.split("\t");
			        name = values[3];
			        break;
			        
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		return name;
	}

	static String mapTargetCUItoEnsembl(String cui, String ddiPath) {
		
   		String id="null";
		String line;
		String drugMappings = ddiPath+"target-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
					
			        String[] values = line.split("\t");
			        id = values[4];
			        break;
			        
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		return id;
	}
	static String mapTargetEnsembltoStringId(String gene, String drugMappings) {
		
   		String id="null";
		String line;
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains("\t"+gene+"\t")) {
					
			        String[] values = line.split("\t");
			        id = values[0];
			        break;
			        
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		return id;
	}
				
				
	static String mapCUItoName(String cui, String ddiPath) {
		
   		String name="null";
		String line;
		String drugMappings = ddiPath+"drug-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
					
			        String[] values = line.split("\t");
			        name = values[1];
			        break;
			        
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return name;
	}
	
	
	static String mapCUItoChemBL(String cui, String ddiPath) {
		
   		String chembl="null";
		String line;
		String drugMappings = ddiPath+"drug-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(cui)) {
					
			        String[] values = line.split("\t");
			        chembl = values[5];
			        break;
			        
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return chembl;
	}

	static String getNameMapping(String name, String ddiPath) {
		
   		String cuis="null";
		String line;
		String drugMappings = ddiPath+"drug-mappings_latest.tsv";
		try {
			BufferedReader br = new BufferedReader(new FileReader(drugMappings));
	   		
			while ((line = br.readLine()) != null ){
				if(line.contains(name)) {
					
			        String[] values = line.split("\t");
			        if (name.equals(values[1])) {
			        	cuis = values[11];
			        	break;
			        }
				}    
			}
			br.close();
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return cuis;
	}

	public static LinkedList<String> returnDBpairs (String cui, String ddiPath) {
		
		LinkedList<String> drugs = new LinkedList<String>();
		try {

			
			BufferedReader br = new BufferedReader(new FileReader(ddiPath+"/FeatureExtraction/approachb-features-positivePairs.csv"));
		   	String line;
			
		   	while ((line = br.readLine()) != null ){
				if (line.contains(cui)) {
					String [] values = line.split(",");
					String pair = values[values.length-1] ;
					String []cuis = pair.split("_");
					if (cui.equals(cuis[0]))
						drugs.add(cuis[1]);
					else
						drugs.add(cuis[0]);
				}	
			}		
		   	br.close();
		   	
		}catch(Exception e) {
			e.printStackTrace();
		}
									
		return drugs;
	}


	public static LinkedList<String[]> returnPredictedpairs (String cui, String ddiPath) {
		
		LinkedList<String[]> drugs = new LinkedList<String[]>();
		try {

			
			BufferedReader br = new BufferedReader(new FileReader(ddiPath+"/LC_DDIs/POS_PREDICTIONS.txt"));
		   	String line;
			
		   	while ((line = br.readLine()) != null ){
				if (line.contains(cui)) {
					String [] entry  = new String[2];
					
					String [] values = line.split(",");
					String pair = values[0] ;
					String []cuis = pair.split("_");
					entry[1]= values[1] ;
					if (cui.equals(cuis[0]))
						entry[0]=cuis[1];
					else
						entry[0]=cuis[0];
					drugs.add(entry);
				}	
			}		
		   	br.close();
		   	
		}catch(Exception e) {
			e.printStackTrace();
		}
									
		return drugs;
	}

	public static boolean existsInDeduced (String pair, String ddiPredictionsFile) {
		
		String [] values = pair.split("_");
		String oppositePair = values[1]+","+values[0];
		pair = values[0]+","+values[1];
		boolean exists = false;
		
		try {

			
			BufferedReader br = new BufferedReader(new FileReader(ddiPredictionsFile));
		   	String line;
			
		   	while ((line = br.readLine()) != null ){
				if ((line.contains(pair)) || (line.contains(oppositePair))) {
					exists = true;
					break;
				}
			}
			br.close();
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return exists;
	}
		
	static boolean foundInInterDecagon_targets(String drug , String gene,  String snap2File) {
		boolean found=false;
		try {

			
			BufferedReader br = new BufferedReader(new FileReader(snap2File));
		   	String line;
			
		   	while ((line = br.readLine()) != null ){
				if (line.contains(drug+","+gene)) {
					found = true;
					break;
				}
			}
			br.close();
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return found;
	}

	static boolean foundInChGMiner(String drug , String gene,  String snap1File) {
		boolean found=false;
		try {

			
			BufferedReader br = new BufferedReader(new FileReader(snap1File));
		   	String line;
			
		   	while ((line = br.readLine()) != null ){
				if (line.contains(drug+"\t"+gene)) {
					found = true;
					break;
				}
			}
			br.close();
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return found;
	}

	static boolean foundInOpentargets(String gene,String chembl, String opentargetsFile) {
		boolean found=false;
		try {

			
			BufferedReader br = new BufferedReader(new FileReader(opentargetsFile));
		   	String line;
		   	//System.out.println("Searching Open targets gene "+gene);
			
		   	while ((line = br.readLine()) != null ){
				if (line.contains("\t"+gene+"\t")) {
					String lookForId = "'"+chembl+"'";
					//System.out.println("Found in Open targets gene "+gene+", look for id="+lookForId);
					
					if(line.contains(lookForId))
						found = true;
					break;
					
				}
			}
			br.close();
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return found;
	}
	
	static boolean foundbindingDB(String uniid,String bindid, String bindDB) {
		boolean found=false;
		try {

			
			BufferedReader br = new BufferedReader(new FileReader(bindDB));
		   	String line;
		   	//System.out.println("Searching Open targets gene "+gene);
			
		   	while ((line = br.readLine()) != null ){
				if (line.contains("\t"+uniid+"\t")) {
					String lookForId = "\t"+bindid+"\t";
					//System.out.println("Found in bindingDB protein "+uniid+", look for id="+lookForId);
					
					if(line.contains(lookForId)) {
						found = true;
						break;
					}	
				}
			}
			br.close();
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return found;
	}
	
	static boolean foundInStitch(String stringId,String drug, String bindDB) {
		boolean found=false;
		try {

			
			BufferedReader br = new BufferedReader(new FileReader(bindDB));
		   	String line;
		   	System.out.println("Searching STITCH for "+stringId+ " "+ drug);
			
		   	while ((line = br.readLine()) != null ){
				if (line.contains(drug+"\t")) {
					String lookForId = "\t"+stringId+"\t";
					//System.out.println("Found in stitch drug "+uniid+", look for id="+lookForId);
					
					if(line.contains(lookForId)) {
						found = true;
						break;
					}	
				}
			}
			br.close();
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return found;
	}
	
	static boolean foundInCTD(String gene,String drName,String cas_num, String ctdFile)  {
		boolean found=false;
		try {

			
			BufferedReader br = new BufferedReader(new FileReader(ctdFile));
		   	String line;
		   	System.out.println("Searching CTD for "+drName+ " "+ gene);
			
		   	while ((line = br.readLine()) != null ){
				if ((line.contains(drName+"\t")) || (line.contains("\t"+cas_num+"\t"))) {
					String lookForId = "\t"+gene+"\t";
					//System.out.println("Found in stitch drug "+uniid+", look for id="+lookForId);
					
					if(line.contains(lookForId)) {
						found = true;
						break;
					}	
				}
			}
			br.close();
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return found;
	}
	
	static boolean foundInDrugCentral(String uniname, String drug, String dcfile) {
		boolean found=false;
		try {
			
			BufferedReader br = new BufferedReader(new FileReader(dcfile));
		   	String line;
			
		   	while ((line = br.readLine()) != null ){
		   		String [] values = line.split("\t");
				if (values[0].equalsIgnoreCase("\""+drug+"\"")) { //found drug section
					//System.out.println("In DrugCentral "+drug+ ". Looking for "+ uniname);
					if(line.contains(uniname)) {
							found = true;
							break;
					}
					
				}
			}
			br.close();
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return found;
	}
	
	static boolean existsInPosGroundtruth(String dbid, String gene, String posground) {
		boolean found=false;
		
		try {
			
			BufferedReader br = new BufferedReader(new FileReader(posground));
		   	String line;
			
		   	while ((line = br.readLine()) != null ){
		   		
				if (line.contains(dbid+","+gene)) { //found!!! shouldnt be there
						found = true;
						break;
				}
			}
			br.close();
			
		}catch(Exception e) {
			e.printStackTrace();
		}
		return found;
	}
}
