import java.io.PrintWriter;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;

import java.util.List;
import java.util.Map.Entry;
import java.util.Set;

/**
 * How to sort HashMap in Java by keys and values. 
 * HashMap doesn't guarantee any order, so you cannot rely on it, even if
 * it appear that it storing entries in a particular order, because
 * it may not be available in future version e.g. earlier HashMap stores
 * integer keys on the order they are inserted but from Java 8 it has changed.
 * 
 * @author WINDOWS 8
 */

public class HashMapSorting{

    public static void sortAndSave(int n, PrintWriter writer, HashMap<String, Integer> codenames) throws ParseException {
        
        
        System.out.println("HashMap before sorting, random order ");
        Set<Entry<String, Integer>> entries = codenames.entrySet();
       
        // Now let's sort the HashMap by values
        // there is no direct way to sort HashMap by values but you
        // can do this by writing your own comparator, which takes
        // Map.Entry object and arrange them in order increasing 
        // or decreasing by values.
        
        Comparator<Entry<String, Integer>> valueComparator 
               = new Comparator<Entry<String,Integer>>() {
            
            @Override
            public int compare(Entry<String, Integer> e1, Entry<String, Integer> e2) {
                Integer v1 = e1.getValue();
                Integer v2 = e2.getValue();
                return v1.compareTo(v2);
            }
        };
        
        // Sort method needs a List, so let's first convert Set to List in Java
        List<Entry<String, Integer>> listOfEntries 
                  = new ArrayList<Entry<String, Integer>>(entries);
        
        // sorting HashMap by values using comparator
        Collections.sort(listOfEntries, valueComparator);
        Collections.reverse(listOfEntries);
        
         int i=0;
        // copying entries from List to Map
        for(Entry<String, Integer> entry : listOfEntries){
           
        	if (++i>n)
        		break;
            System.out.println(entry.getKey() + " ==> " + entry.getValue());
            writer.println(entry.getKey());
        }
    }
    
}