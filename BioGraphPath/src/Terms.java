import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;


public class Terms {

	public List<String> relationTypes = new ArrayList<String>();
	public List<String> nonNormrelationTypes = new ArrayList<String>();
	public HashMap<String,String> normalizedNames= new HashMap<String, String>();
	
	public Terms() {

		 AddRelationCodes();
		 addNormalizedNames();
		 AddNonNormalizedRelationCodes();
		 
		 //currently not needed...
		// AddDrugs(drugsListFile);
		 
	}
	
	
	void AddRelationCodes() {
			
		relationTypes.add("ADMINISTERED_TO");
		relationTypes.add("AFFECTS");
		relationTypes.add("ASSOCIATED_WITH");
		relationTypes.add("AUGMENTS");
		relationTypes.add("CAUSES");
		relationTypes.add("COEXISTS_WITH");
		relationTypes.add("compared_with");
		relationTypes.add("COMPLICATES");
		relationTypes.add("CONVERTS_TO");
		relationTypes.add("DIAGNOSES");
		relationTypes.add("different_from");
		relationTypes.add("different_than");
		relationTypes.add("DISRUPTS");
		relationTypes.add("higher_than");
		relationTypes.add("INHIBITS");
		relationTypes.add("INTERACTS_WITH");
		relationTypes.add("IS_A");
		relationTypes.add("ISA");
		relationTypes.add("LOCATION_OF");
		relationTypes.add("lower_than");
		relationTypes.add("MANIFESTATION_OF");
		relationTypes.add("METHOD_OF");
		relationTypes.add("OCCURS_IN");
		relationTypes.add("PART_OF");
		relationTypes.add("PRECEDES");
		relationTypes.add("PREDISPOSES");
		relationTypes.add("PREVENTS");
		relationTypes.add("PROCESS_OF");
		relationTypes.add("PRODUCES");
		relationTypes.add("same_as");
		relationTypes.add("STIMULATES");
		relationTypes.add("TREATS");
		relationTypes.add("USES");
		relationTypes.add("MENTIONED_IN");
		relationTypes.add("HAS_MESH");
		relationTypes.add("LITERATURE_DTI");
		
	}
	
	
	void AddNonNormalizedRelationCodes() {
			
		nonNormrelationTypes.add("ADMINISTERED_TO");//"ADMINISTERED_TO");
		nonNormrelationTypes.add("ADMINISTERED_TO__SPEC__");// "ADMINISTERED_TO");
		nonNormrelationTypes.add("AFFECTS");// "AFFECTS");
		nonNormrelationTypes.add("AFFECTS__SPEC__");// "AFFECTS");
		nonNormrelationTypes.add("ASSOCIATED_WITH");// "ASSOCIATED_WITH");
		nonNormrelationTypes.add("ASSOCIATED_WITH__INFER__");// "ASSOCIATED_WITH");
		nonNormrelationTypes.add("ASSOCIATED_WITH__SPEC__");// "ASSOCIATED_WITH");
		nonNormrelationTypes.add("AUGMENTS");// "AUGMENTS");
		nonNormrelationTypes.add("AUGMENTS__SPEC__");// "AUGMENTS");
		nonNormrelationTypes.add("CAUSES");// "CAUSES");
		nonNormrelationTypes.add("CAUSES__SPEC__");// "CAUSES");
		nonNormrelationTypes.add("COEXISTS_WITH");// "COEXISTS_WITH");
		nonNormrelationTypes.add("COEXISTS_WITH__SPEC__");// "COEXISTS_WITH");
		nonNormrelationTypes.add("compared_with");// "compared_with");
		nonNormrelationTypes.add("compared_with__SPEC__");// "compared_with");
		nonNormrelationTypes.add("COMPLICATES");// "COMPLICATES");
		nonNormrelationTypes.add("COMPLICATES__SPEC__");// "COMPLICATES");
		nonNormrelationTypes.add("CONVERTS_TO");// "CONVERTS_TO");
		nonNormrelationTypes.add("CONVERTS_TO__SPEC__");// "CONVERTS_TO");
		nonNormrelationTypes.add("DIAGNOSES");// "DIAGNOSES");
		nonNormrelationTypes.add("DIAGNOSES__SPEC__");// "DIAGNOSES");
		nonNormrelationTypes.add("different_from");// "different_from");
		nonNormrelationTypes.add("different_from__SPEC__");// "different_from");
		nonNormrelationTypes.add("different_than");// "different_than");
		nonNormrelationTypes.add("different_than__SPEC__");// "different_than");
		nonNormrelationTypes.add("DISRUPTS");// "DISRUPTS");
		nonNormrelationTypes.add("DISRUPTS__SPEC__");// "DISRUPTS");
		nonNormrelationTypes.add("higher_than");// "higher_than");
		nonNormrelationTypes.add("higher_than__SPEC__");// "higher_than");
		nonNormrelationTypes.add("INHIBITS");// "INHIBITS");
		nonNormrelationTypes.add("INHIBITS__SPEC__");// "INHIBITS");
		nonNormrelationTypes.add("INTERACTS_WITH");// "INTERACTS_WITH");
		nonNormrelationTypes.add("INTERACTS_WITH__INFER__");// "INTERACTS_WITH");
		nonNormrelationTypes.add("INTERACTS_WITH__SPEC__");// "INTERACTS_WITH");
		nonNormrelationTypes.add("IS_A");// "IS_A");
		nonNormrelationTypes.add("ISA");// "ISA");
		nonNormrelationTypes.add("LOCATION_OF");// "LOCATION_OF");
		nonNormrelationTypes.add("LOCATION_OF__SPEC__");// "LOCATION_OF");
		nonNormrelationTypes.add("lower_than");// "lower_than");
		nonNormrelationTypes.add("lower_than__SPEC__");// "lower_than");
		nonNormrelationTypes.add("MANIFESTATION_OF");// "MANIFESTATION_OF");
		nonNormrelationTypes.add("MANIFESTATION_OF__SPEC__");// "MANIFESTATION_OF");
		nonNormrelationTypes.add("METHOD_OF");// "METHOD_OF");
		nonNormrelationTypes.add("METHOD_OF__SPEC__");// "METHOD_OF");
		nonNormrelationTypes.add("OCCURS_IN");// "OCCURS_IN");
		nonNormrelationTypes.add("OCCURS_IN__SPEC__");// "OCCURS_IN");
		nonNormrelationTypes.add("PART_OF");// "PART_OF");
		nonNormrelationTypes.add("PART_OF__SPEC__");// "PART_OF");
		nonNormrelationTypes.add("PRECEDES");// "PRECEDES");
		nonNormrelationTypes.add("PRECEDES__SPEC__");// "PRECEDES");
		nonNormrelationTypes.add("PREDISPOSES");// "PREDISPOSES");
		nonNormrelationTypes.add("PREDISPOSES__SPEC__");// "PREDISPOSES");
		nonNormrelationTypes.add("PREVENTS");// "PREVENTS");
		nonNormrelationTypes.add("PREVENTS__SPEC__");// "PREVENTS");
		nonNormrelationTypes.add("PROCESS_OF");// "PROCESS_OF");
		nonNormrelationTypes.add("PROCESS_OF__SPEC__");// "PROCESS_OF");
		nonNormrelationTypes.add("PRODUCES");// "PRODUCES");
		nonNormrelationTypes.add("PRODUCES__SPEC__");// "PRODUCES");
		nonNormrelationTypes.add("same_as");// "same_as");
		nonNormrelationTypes.add("same_as__SPEC__");// "same_as");
		nonNormrelationTypes.add("STIMULATES");// "STIMULATES");
		nonNormrelationTypes.add("STIMULATES__SPEC__");// "STIMULATES");
		nonNormrelationTypes.add("TREATS");// "TREATS");
		nonNormrelationTypes.add("TREATS__INFER__");// "TREATS");
		nonNormrelationTypes.add("TREATS__SPEC__");// "TREATS");
		nonNormrelationTypes.add("USES");// "USES");
		nonNormrelationTypes.add("USES__SPEC__");// "USES");
		nonNormrelationTypes.add("MENTIONED_IN");
		nonNormrelationTypes.add("HAS_MESH");
		nonNormrelationTypes.add("LITERATURE_DTI");
		
	}

	
	void addNormalizedNames() {
		normalizedNames.put("ADMINISTERED_TO","ADMINISTERED_TO");
		normalizedNames.put("ADMINISTERED_TO__SPEC__", "ADMINISTERED_TO");
		normalizedNames.put("AFFECTS", "AFFECTS");
		normalizedNames.put("AFFECTS__SPEC__", "AFFECTS");
		normalizedNames.put("ASSOCIATED_WITH", "ASSOCIATED_WITH");
		normalizedNames.put("ASSOCIATED_WITH__INFER__", "ASSOCIATED_WITH");
		normalizedNames.put("ASSOCIATED_WITH__SPEC__", "ASSOCIATED_WITH");
		normalizedNames.put("AUGMENTS", "AUGMENTS");
		normalizedNames.put("AUGMENTS__SPEC__", "AUGMENTS");
		normalizedNames.put("CAUSES", "CAUSES");
		normalizedNames.put("CAUSES__SPEC__", "CAUSES");
		normalizedNames.put("COEXISTS_WITH", "COEXISTS_WITH");
		normalizedNames.put("COEXISTS_WITH__SPEC__", "COEXISTS_WITH");
		normalizedNames.put("compared_with", "compared_with");
		normalizedNames.put("compared_with__SPEC__", "compared_with");
		normalizedNames.put("COMPLICATES", "COMPLICATES");
		normalizedNames.put("COMPLICATES__SPEC__", "COMPLICATES");
		normalizedNames.put("CONVERTS_TO", "CONVERTS_TO");
		normalizedNames.put("CONVERTS_TO__SPEC__", "CONVERTS_TO");
		normalizedNames.put("DIAGNOSES", "DIAGNOSES");
		normalizedNames.put("DIAGNOSES__SPEC__", "DIAGNOSES");
		normalizedNames.put("different_from", "different_from");
		normalizedNames.put("different_from__SPEC__", "different_from");
		normalizedNames.put("different_than", "different_than");
		normalizedNames.put("different_than__SPEC__", "different_than");
		normalizedNames.put("DISRUPTS", "DISRUPTS");
		normalizedNames.put("DISRUPTS__SPEC__", "DISRUPTS");
		normalizedNames.put("higher_than", "higher_than");
		normalizedNames.put("higher_than__SPEC__", "higher_than");
		normalizedNames.put("INHIBITS", "INHIBITS");
		normalizedNames.put("INHIBITS__SPEC__", "INHIBITS");
		normalizedNames.put("INTERACTS_WITH", "INTERACTS_WITH");
		normalizedNames.put("INTERACTS_WITH__INFER__", "INTERACTS_WITH");
		normalizedNames.put("INTERACTS_WITH__SPEC__", "INTERACTS_WITH");
		normalizedNames.put("IS_A", "IS_A");
		normalizedNames.put("ISA", "ISA");
		normalizedNames.put("LOCATION_OF", "LOCATION_OF");
		normalizedNames.put("LOCATION_OF__SPEC__", "LOCATION_OF");
		normalizedNames.put("lower_than", "lower_than");
		normalizedNames.put("lower_than__SPEC__", "lower_than");
		normalizedNames.put("MANIFESTATION_OF", "MANIFESTATION_OF");
		normalizedNames.put("MANIFESTATION_OF__SPEC__", "MANIFESTATION_OF");
		normalizedNames.put("METHOD_OF", "METHOD_OF");
		normalizedNames.put("METHOD_OF__SPEC__", "METHOD_OF");
		normalizedNames.put("OCCURS_IN", "OCCURS_IN");
		normalizedNames.put("OCCURS_IN__SPEC__", "OCCURS_IN");
		normalizedNames.put("PART_OF", "PART_OF");
		normalizedNames.put("PART_OF__SPEC__", "PART_OF");
		normalizedNames.put("PRECEDES", "PRECEDES");
		normalizedNames.put("PRECEDES__SPEC__", "PRECEDES");
		normalizedNames.put("PREDISPOSES", "PREDISPOSES");
		normalizedNames.put("PREDISPOSES__SPEC__", "PREDISPOSES");
		normalizedNames.put("PREVENTS", "PREVENTS");
		normalizedNames.put("PREVENTS__SPEC__", "PREVENTS");
		normalizedNames.put("PROCESS_OF", "PROCESS_OF");
		normalizedNames.put("PROCESS_OF__SPEC__", "PROCESS_OF");
		normalizedNames.put("PRODUCES", "PRODUCES");
		normalizedNames.put("PRODUCES__SPEC__", "PRODUCES");
		normalizedNames.put("same_as", "same_as");
		normalizedNames.put("same_as__SPEC__", "same_as");
		normalizedNames.put("STIMULATES", "STIMULATES");
		normalizedNames.put("STIMULATES__SPEC__", "STIMULATES");
		normalizedNames.put("TREATS", "TREATS");
		normalizedNames.put("TREATS__INFER__", "TREATS");
		normalizedNames.put("TREATS__SPEC__", "TREATS");
		normalizedNames.put("USES", "USES");
		normalizedNames.put("USES__SPEC__", "USES");
		normalizedNames.put("MENTIONED_IN", "MENTIONED_IN");
		normalizedNames.put("HAS_MESH", "HAS_MESH");
		normalizedNames.put("LITERATURE_DTI","LITERATURE_DTI");
	}
}
