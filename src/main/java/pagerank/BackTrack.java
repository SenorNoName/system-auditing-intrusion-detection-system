package pagerank;
//import com.sun.org.apache.xpath.internal.SourceTree;
import org.jgrapht.alg.CycleDetector;
import org.jgrapht.graph.DirectedPseudograph;

import pagerank.EntityNode;
import pagerank.EventEdge;
import pagerank.IterateGraph;

import java.math.BigDecimal;
import java.util.*;

/**
 * Created by fang on 3/23/18.
 * the input is the original graph
 */
public class BackTrack {
    DirectedPseudograph<EntityNode,EventEdge> originalGraph;
    IterateGraph iterateGraph;
    DirectedPseudograph<EntityNode, EventEdge> afterBackTrack;
    private Map<String, Integer> stepInfo;
    private CycleDetector<EntityNode, EventEdge> cycleDetector;

    BackTrack(DirectedPseudograph<EntityNode, EventEdge> input){
        originalGraph = (DirectedPseudograph<EntityNode, EventEdge>)input.clone();
        iterateGraph = new IterateGraph(originalGraph);
        stepInfo = new HashMap<>();
        cycleDetector = new CycleDetector<>(originalGraph);
    }
    //Fang's comment this backtrack fixed all the existing issue, recommend using this one
    public DirectedPseudograph<EntityNode, EventEdge> backTrackPOIEvent(String str){
        System.out.println("backTrackPOIEvent invoked: "+str);
        DirectedPseudograph<EntityNode, EventEdge> backTrack = new DirectedPseudograph<EntityNode, EventEdge>(EventEdge.class);
        EntityNode start = iterateGraph.getGraphVertex(str);
        BigDecimal latestOPTime = iterateGraph.getLatestOperationTime(start);
        Map<EntityNode,BigDecimal> timeThresolds = new HashMap<>();
        timeThresolds.put(start, latestOPTime);
        Set<EntityNode> nodeInTheQueue = new HashSet<>();
        Queue<EntityNode> queue = new LinkedList<>();
        nodeInTheQueue.add(start);
        queue.offer(start);

        while(!queue.isEmpty()){
            EntityNode cur = queue.poll();
            backTrack.addVertex(cur);
            Set<EventEdge> incoming = originalGraph.incomingEdgesOf(cur);
            BigDecimal curThresold = timeThresolds.get(cur);
            for(EventEdge e:incoming){
                if(e.getStartTime().compareTo(curThresold)>0) continue;
                EntityNode source = e.getSource();
                backTrack.addVertex(source);
                timeThresolds.putIfAbsent(source, BigDecimal.ZERO);
                BigDecimal thresoldForSource = e.endTime.compareTo(curThresold)<0? e.endTime:curThresold;
                if(timeThresolds.get(source).compareTo(thresoldForSource) < 0){
                    timeThresolds.put(source, thresoldForSource);
                }
                backTrack.addEdge(source, cur, e);
                if(!nodeInTheQueue.contains(source)){
                    nodeInTheQueue.add(source);
                    queue.offer(source);
                }
            }
        }
        afterBackTrack = backTrack;
        return backTrack;
    }

    //Fang's comment this backtrack fixed all the existing issue, recommend using this one
    public DirectedPseudograph<EntityNode, EventEdge> backTrackPOIEventWithStep(String str){
        System.out.println("backTrackPOIEvent invoked: "+str);
        DirectedPseudograph<EntityNode, EventEdge> backTrack = new DirectedPseudograph<EntityNode, EventEdge>(EventEdge.class);
        EntityNode start = iterateGraph.getGraphVertex(str);
        BigDecimal latestOPTime = iterateGraph.getLatestOperationTime(start);
        Map<EntityNode,BigDecimal> timeThresolds = new HashMap<>();
        timeThresolds.put(start, latestOPTime);
        Set<EntityNode> nodeInTheQueue = new HashSet<>();
        Queue<EntityNode> queue = new LinkedList<>();
        nodeInTheQueue.add(start);
        queue.offer(start);
        int step = 1;
        stepInfo.put(start.getSignature(), step);

        while(!queue.isEmpty()){
            EntityNode cur = queue.poll();
            backTrack.addVertex(cur);
            Set<EventEdge> incoming = originalGraph.incomingEdgesOf(cur);
            BigDecimal curThresold = timeThresolds.get(cur);
            for(EventEdge e:incoming){
                if(e.getStartTime().compareTo(curThresold)>0) continue;
                EntityNode source = e.getSource();
                backTrack.addVertex(source);
                if(!stepInfo.containsKey(source.getSignature())){
                    stepInfo.put(source.getSignature(), stepInfo.get(cur.getSignature())+1);
                }else{
                    Set<EntityNode> cycleContainsSource = cycleDetector.findCyclesContainingVertex(source);
                    if(!cycleContainsSource.contains(cur)){
                        step = Math.max(stepInfo.get(source.getSignature()), stepInfo.get(cur.getSignature())+1);
                        stepInfo.put(source.getSignature(), step);
                    }
                }
                timeThresolds.putIfAbsent(source, BigDecimal.ZERO);
                BigDecimal thresoldForSource = e.endTime.compareTo(curThresold)<0? e.endTime:curThresold;
                if(timeThresolds.get(source).compareTo(thresoldForSource) < 0){
                    timeThresolds.put(source, thresoldForSource);
                }
                backTrack.addEdge(source, cur, e);
                if(!nodeInTheQueue.contains(source)){
                    nodeInTheQueue.add(source);
                    queue.offer(source);
                }
            }
        }
        afterBackTrack = backTrack;
        return backTrack;
    }

    // Fang'comment: method strictly follow the paper
    public DirectedPseudograph<EntityNode, EventEdge> backTrackPOIEvent2(String str){

        DirectedPseudograph<EntityNode, EventEdge> backTrack = new DirectedPseudograph<EntityNode, EventEdge>(EventEdge.class);
        EntityNode start = iterateGraph.getGraphVertex(str);
        BigDecimal latestOPTime = iterateGraph.getLatestOperationTime(start);
        Map<EntityNode, BigDecimal> timeThresold = new HashMap<>();
        timeThresold.put(start, latestOPTime);
        List<EventEdge> edgeList = new LinkedList<>(originalGraph.edgeSet());
        Collections.sort(edgeList, (a, b)->b.getStartTime().compareTo(a.getStartTime()));
        backTrack.addVertex(start);

        for(EventEdge e: edgeList){
            EntityNode source = e.getSource();
            EntityNode target = e.getSink();
            if(backTrack.containsVertex(target)){
                BigDecimal targetThresold = timeThresold.get(target);
                if(e.getStartTime().compareTo(targetThresold)<0) {
                    if (!backTrack.containsVertex(source)) {
                        backTrack.addVertex(source);
                        BigDecimal thresoldForObject = e.getEndTime().compareTo(targetThresold)<0? e.getEndTime():targetThresold;
                        timeThresold.put(source, thresoldForObject);
                    }
                    backTrack.addEdge(source,target,e);
                }
            }
        }
        afterBackTrack = backTrack;
        return backTrack;

    }

    public void printGraph(){
        assert afterBackTrack != null;
        IterateGraph iter = new IterateGraph(afterBackTrack);
        iter.exportGraph("backtrackTest");
    }

    public void exportGraph(String file){
        assert afterBackTrack != null;
        IterateGraph iter = new IterateGraph(afterBackTrack);
        iter.exportGraph(file);
    }

    public Map<String, Integer> getBackwardStepInfo(){
        return Collections.unmodifiableMap(stepInfo);
    }


}
