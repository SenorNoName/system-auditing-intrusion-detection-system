package pagerank;

import org.jgrapht.graph.DirectedPseudograph;


import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.*;

/**
 * Created by fang on 4/6/18.
 */

public class ProcessOneLog {

    public static void main(String[] args){
        ArgParser ap = new ArgParser(args);
        Map<String,String> argMap = ap.parseArgs();
//        String path = "/home/lcl/logs/file_manipulation/1.txt";
        String[] localIP = MetaConfig.localIP;
//        String detection = "/home/lcl/wget-1.19/INSTALL";
//        String[] highRP = {};
        String[] midRP = MetaConfig.midRP;//        String[] lowRP = {"192.168.29.234:54764->208.118.235.20:80"};//"/media/lcl/LCL/bad.zip"};//"192.168.29.125:10289->192.168.29.234:22"}
        String path = argMap.get("path");
        String detection = argMap.get("detection");
        String highRPs = argMap.get("high");
        String[] highRP = highRPs==null?new String[]{}:highRPs.split(",");
        String neutralRPs = argMap.get("neutral");
        String[] neutralRP = neutralRPs==null?new String[]{}:neutralRPs.split(",");
        ArrayList<String> midRP2 = new ArrayList<>();
        midRP2.addAll(Arrays.asList(midRP));
        midRP2.addAll(Arrays.asList(neutralRP));
        String lowRPs = argMap.get("low");
        String[] lowRP = lowRPs==null?new String[]{}:lowRPs.split(",");
        String resultDir = argMap.get("res");
        String suffix = argMap.get("suffix");
        double threshold = Double.parseDouble(argMap.get("thresh"));
        boolean trackOrigin = argMap.containsKey("origin");

        String[] paths = path.split("/");
        process(resultDir, suffix, threshold, trackOrigin, path,localIP,detection,highRP,midRP2.toArray(new String[midRP2.size()]),lowRP, paths[paths.length-1],0, new HashSet<>(),null);
    }

    public static void process(String resultDir, String suffix, double threshold, boolean trackOrigin, String logfile, String[] IP, String detection,String[] highRP,String[] midRP,String[] lowRP, String filename,double detectionSize, Set<String> seedSources,String[] criticalEdges){
        //String resultDir =  "/home/lcl/results/exp/";
        //String suffix = "";
        OutputStream os = null;
        OutputStream weightfile = null;
        try{
            long start = System.currentTimeMillis();
            File result_folder = new File(resultDir);
            if (!result_folder.exists()) {
                result_folder.mkdir();
            }
            String explogfile = resultDir + File.separator + filename + suffix + "_stats";
            os = new FileOutputStream(explogfile);
            GetGraph getGraph = new GetGraph(logfile, IP);
            getGraph.GenerateGraph();
            DirectedPseudograph<EntityNode, EventEdge> orignal = getGraph.getJg();


            long end = System.currentTimeMillis();
            System.out.println("Parsing time:"+(end-start)/1000.0);
            System.out.println("Original vertex number:" + orignal.vertexSet().size() + " edge number : " + orignal.edgeSet().size());
            os.write(("Original vertex number:" + orignal.vertexSet().size() + " edge number : " + orignal.edgeSet().size()+"\n").getBytes());

            start = System.currentTimeMillis();
            BackTrack backTrack = new BackTrack(orignal);
            backTrack.backTrackPOIEvent(detection);
            end = System.currentTimeMillis();
            System.out.println("Backtrack time:"+(end-start)/1000.0);
            System.out.println("After Backtrack vertex number is: "+ backTrack.afterBackTrack.vertexSet().size() + " edge number: " + backTrack.afterBackTrack.edgeSet().size());
            os.write(("After Backtrack vertex number is: "+ backTrack.afterBackTrack.vertexSet().size() + " edge number: " + backTrack.afterBackTrack.edgeSet().size()+"\n").getBytes());

            IterateGraph out = new IterateGraph(backTrack.afterBackTrack);
            out.exportGraph(resultDir+"BackTrack_"+filename+suffix);
//            backTrack.exportGraph("backTrack");

//            // If we run cpr1 and cpr2 together, the cpr1 result will effect cpr2, need more investigation
//            CausalityPreserve CPR1 = new CausalityPreserve(backTrack.afterBackTrack);
//            CPR1.CPR(1);
//            System.out.println("After CPR1 vertex number is: "+ CPR1.afterMerge.vertexSet().size() + " edge number: " + CPR1.afterMerge.edgeSet().size());
//            os.write(("After CPR1 vertex number is: "+ CPR1.afterMerge.vertexSet().size() + " edge number: " + CPR1.afterMerge.edgeSet().size()+"\n").getBytes());

            start = System.currentTimeMillis();
            CausalityPreserve CPR = new CausalityPreserve(backTrack.afterBackTrack);
            CPR.mergeEdgeFallInTheRange2(10.0);
            end = System.currentTimeMillis();
            System.out.println("CPR time:"+(end-start)/1000.0);
            System.out.println("After CPR vertex number is: "+ CPR.afterMerge.vertexSet().size() + " edge number: " + CPR.afterMerge.edgeSet().size());
            os.write(("After CPR vertex number is: "+ CPR.afterMerge.vertexSet().size() + " edge number: " + CPR.afterMerge.edgeSet().size()+"\n").getBytes());

            out = new IterateGraph(CPR.afterMerge);
            out.exportGraph(resultDir+"AfterCPR_"+filename+suffix);
            start = System.currentTimeMillis();
            GraphSplit split = new GraphSplit(CPR.afterMerge);
            split.splitGraph();
            end = System.currentTimeMillis();
            System.out.println("split time:"+(end-start)/1000.0);
            System.out.println("After Split vertex number is: "+ split.inputGraph.vertexSet().size() + " edge number: " + split.inputGraph.edgeSet().size());

            InferenceReputation infer = new InferenceReputation(split.inputGraph);

            os.write(("After Split vertex number is: "+ split.inputGraph.vertexSet().size() + " edge number: " + split.inputGraph.edgeSet().size()+"\n").getBytes());
//            weightfile = new FileOutputStream(resultDir+"weights_"+filename+suffix);
//            for(EventEdge e: infer.graph.edgeSet()){
//                weightfile.write((String.valueOf(e.weight)+",").getBytes());
//            }

            infer.setDetectionSize(detectionSize);
            infer.setSeedSources(seedSources);   // set structure weight for source
            start = System.currentTimeMillis();
            infer.calculateWeights_ML(true,3);
//            infer.calculateWeights_Individual(true, "timeWeight");
//            infer.calculateWeights();
            //System.out.println("OTSU: "+infer.OTSUThreshold());
            end = System.currentTimeMillis();
            System.out.println("Weight computation time:"+(end-start)/1000.0);


            //infer.normalizeWeightsAfterFiltering();
            infer.initialReputation(highRP,midRP,lowRP);
            start = System.currentTimeMillis();
            infer.PageRankIteration2(highRP,midRP,lowRP,detection);
            end = System.currentTimeMillis();
            double timeCost = (end-start)*1.0/1000.0;
            System.out.println("Reputation propergation time is: "+ timeCost);
            //infer.PageRankIteration(detection);
            //infer.fixReputation(highRP);

//            System.out.println("After Filter vertex number is: "+ split.inputGraph.vertexSet().size() + " edge number: " + split.inputGraph.edgeSet().size());
//            os.write(("After Filter vertex number is: "+ split.inputGraph.vertexSet().size() + " edge number: " + split.inputGraph.edgeSet().size()).getBytes());

//        //infer.onlyPrintHighestWeights(detection);
            infer.removeIrrelaventVertices(detection);
            infer.exportGraph(resultDir+"Weight_"+filename+suffix);
//        IterateGraph iterGraph = new IterateGraph(infer.graph);
            for(EntityNode v : infer.graph.vertexSet())
                if(v.getSignature().equals(detection)){
                    os.write(("POI Reputation: "+v.getReputation()).getBytes());
                    break;
                }

            Set<String> criticalNodes = new HashSet<>();

                for(String edge : criticalEdges) {
                criticalNodes.add(edge.split(",")[0]);
                criticalNodes.add(edge.split(",")[1]);
            }

            int roots = 0;
            for(EntityNode n : infer.graph.vertexSet()) {
                if(infer.graph.incomingEdgesOf(n).size()==0) {
                    roots++;
                }
            }

            double totalCriticalWeights = 0.0;
            double totalNonCriticalWeights = 0.0;

            double totalCriticalRep = 0.0;
            double totalNonCriticalRep = 0.0;

            int criticalEdgeCount = 0;
            int criticalNodeCount = 0;

	        Set<String> criticalEdgeSet = new HashSet<>(Arrays.asList(criticalEdges));
            for(EventEdge e : infer.graph.edgeSet()) {
            	if(criticalEdgeSet.contains(e.getSource().getSignature()+","+e.getSink().getSignature())) {
                    criticalEdgeCount++;
            	    totalCriticalWeights += e.weight;
	            }else{
            		totalNonCriticalWeights += e.weight;
	            }
            }

            for(EntityNode n : infer.graph.vertexSet()) {
            	if(criticalNodes.contains(n.getSignature())) {
            		criticalNodeCount++;
            	    totalCriticalRep += n.reputation;
	            }else{
            		totalNonCriticalRep += n.reputation;
	            }
            }
            System.out.println("entries:"+roots);

	        System.out.println("#critical edges:"+criticalEdgeCount);
	        System.out.println("#non-critical edges:"+(infer.graph.edgeSet().size()-criticalEdgeCount));

	        System.out.println("#critical nodes:"+criticalNodeCount);
	        System.out.println("#non-critical nodes:"+(infer.graph.vertexSet().size()-criticalNodeCount));


	        System.out.println("avg critical edge weight:"+totalCriticalWeights/criticalEdgeCount);
	        System.out.println("avg non-critical edge weight:"+totalNonCriticalWeights/(infer.graph.edgeSet().size()-criticalEdgeCount));

	        System.out.println("avg critical node rep:"+totalCriticalRep/criticalNodeCount);
	        System.out.println("avg non-critical node rep:"+totalNonCriticalRep/(infer.graph.vertexSet().size()-criticalNodeCount));


//            infer.extractSuspects(0.5);
//            infer.exportGraph(resultDir+"Suspect_"+filename+suffix);
//        iterGraph.filterGraphBasedOnVertexReputation();
//        iterGraph.removeSingleVertex();
//        iterGraph.exportGraph("FilteredInstallMongodb");
//        List<DirectedPseudograph<EntityNode, EventEdge>> paths = iterGraph.getHighWeightPaths(detection);
//        for(int i=0; i< paths.size();i++){
//            IterateGraph iter = new IterateGraph(paths.get(i));
//            String fileName = String.valueOf(i) + "path";
//            iter.exportGraph(fileName);
//        }
            //iterGraph.printEdgesOfVertex("11035dpkg");
//        infer.checkWeightsAfterCalculation();
//        infer.exportGraph("UnrarReputation");
            Runtime rt = Runtime.getRuntime();
//            String[] cmd = {"/bin/sh","-c","dot -T svg "+resultDir+"BackTrack_"+filename+suffix+".dot"
//                    + " > "+resultDir+"BackTrack_"+filename+suffix+".svg"};
//            rt.exec(cmd);
            String[] cmd = new String[] {"/bin/sh","-c","dot -T svg "+resultDir+"AfterCPR_"+filename+suffix+".dot"
                    + " > "+resultDir+"AfterCPR_"+filename+suffix+".svg"};
            rt.exec(cmd);
            cmd = new String[] {"/bin/sh", "-c","dot -T svg "+resultDir+"Weight_"+filename+suffix+".dot"
                    + " > "+resultDir+"Weight_"+filename+suffix+".svg"};
            rt.exec(cmd);

            double avg = infer.getAvgWeight();
            double minCritical = 0.05;
//            Set<String> criticalSet = new HashSet<>(Arrays.asList(criticalEdges));
//            PriorityQueue<EventEdge> pq = new PriorityQueue<>(Comparator.comparing(e -> e.weight));
//
//            Map<String,Double> weightMap = new HashMap<>();
//            for(EventEdge e : infer.graph.edgeSet()){
//                pq.offer(e);
//                weightMap.put(e.getSource().getSignature()+","+e.getSink().getSignature(),e.weight);
//            }

//            List<Double> criticalWeights = new ArrayList<>();
//
//            for(String criticalEdge : criticalEdges){
//                if(!weightMap.containsKey(criticalEdge)){
//                    System.out.println("no key:"+criticalEdge);
//                }
//                double weight1 = weightMap.getOrDefault(criticalEdge,0.0);
//                System.out.println(criticalEdge+":"+weight1);
//                String back = criticalEdge.split(",")[1]+","+criticalEdge.split(",")[0];
//                double weight2 = weightMap.getOrDefault(back,0.0);
//                System.out.println(back+":"+weight2);
//                criticalWeights.add(Math.max(weight1,weight2)/avg);
//                minCritical = Math.min(minCritical,Math.max(weight1,weight2));
//
//            }
//
//            PrintWriter pw = new PrintWriter(resultDir+"/critical_weights");
//            for(double d : criticalWeights)
//                pw.println(d);
//            pw.close();

            double startToMiss = minCritical/avg;

            if(1-minCritical>1e-8){
                infer.filterGraphBasedOnAverageWeight(minCritical);
                infer.removeIsolatedIslands(detection);
            }
            infer.exportGraph(resultDir+"Filter_"+filename+suffix);

            System.out.println(String.format("critical edge number: %d",criticalEdges.length));
            os.write(String.format("\ncritical edge number: %d",criticalEdges.length).getBytes());

            System.out.println(String.format("minimum critical weight: %.3f",minCritical));
            os.write(String.format("\nminimum critical weight: %.3f",minCritical).getBytes());

            System.out.println(String.format("start to miss: %.3f",startToMiss));
            os.write(String.format("\nstart to miss: %.3f",startToMiss).getBytes());

            System.out.println("After Filter vertex number is: "+ infer.graph.vertexSet().size() + " edge number: " + infer.graph.edgeSet().size());
            os.write(String.format("\nAfter Filter vertex number is: %d edge number: %d",infer.graph.vertexSet().size(),infer.graph.edgeSet().size()).getBytes());
            cmd = new String[] {"/bin/sh","-c","dot -T svg "+resultDir+"Filter_"+filename+suffix+".dot"
                    + " > "+resultDir+"Filter_"+filename+suffix+".svg"};
            rt.exec(cmd);
//
//            for(double i=0.0; i<=2.5; i+=0.05){
//                while(pq.size()>0&&pq.peek().weight<i*avg) pq.poll();
//                System.out.println(String.format("%.2f - After Filter edge number is: ",i)+ pq.size());
//                os.write(String.format("\n %.2f - After Filter edge number is: %d",i,pq.size()).getBytes());
//            }


        }catch (Exception e){
            e.printStackTrace();
        }finally {
            try{
                os.close();
//                weightfile.close();
            }catch (Exception e){
                e.printStackTrace();
            }
        }
    }
}
