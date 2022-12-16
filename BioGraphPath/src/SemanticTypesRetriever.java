import java.io.BufferedReader;
import java.io.FileReader;
import java.util.LinkedList;
import java.util.List;

public class SemanticTypesRetriever {

	public String [] getNodeSemTypesArray(String dtifolder) {
		
		String [] semTypes = new String[Enriched_DTI_BLKG .SEMANTIC_TYPES];
	
		try {
			FileReader filerdr = new FileReader(dtifolder+"MetaMap-SemTypes-list.txt");
			BufferedReader in = new BufferedReader(filerdr);
			String line;
			int i=0;
			while(( line = in.readLine() ) != null   ) {
		        String[] elements = line.split("\\|");
		        String semType = elements[0];
		        //System.out.println("Adding sem type "+semType);
				semTypes[i++] = semType;
			}
			in.close();
			filerdr.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return semTypes;
	}
	
	public List<String> getNodeSemTypes (String dtifolder) {
		
		List<String> semTypes = new LinkedList<String> ();
	
		try {
			FileReader filerdr = new FileReader(dtifolder+"MetaMap-SemTypes-list.txt");
			BufferedReader in = new BufferedReader(filerdr);
			String line;

			while(( line = in.readLine() ) != null   ) {
		        String[] elements = line.split("\\|");
		        String semType = elements[0];
		        //System.out.println("Adding sem type "+semType);
				semTypes.add(semType);
			}
			in.close();
			filerdr.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return semTypes;
	}
	public String [] getRelSemTypes(List<String> relationTypes) {
		String [] semTypes = new String[Enriched_DTI_BLKG.RELATIONS];
		int i=0;
		for (String type:relationTypes) {
			semTypes[i++] = type;
		}
		return semTypes;
	}
}
