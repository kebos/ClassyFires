# =========== IMPORTS ============

import copy
import operator
import math
import shelve

# =========== INITIAL VARIABLE VALUES ============

LAMDA=0.8
ITERATIONS=10
PR=True
HITS=True
SAVE_DATABASE=False
VISUALIZE=True


# =========== PAGERANK / HITS FUNCTIONS ============
# Parse the file into a list of lists
def parseData(filename):
    data=[]
    f=open(filename,'r')
    body=f.read()
    lines=body.split("\n")
    for line in lines:
        data.append(line.split(" "))
    f.close()
    return data

# Create the graph = {Node : {pagerank = 1/NODES, received_from:[x,x,x], sent_to:[y,y,y], sent_to_count:z, auth=xx, hub=xx}}
def drawLinks(data):
    graph={}
    for line in data:
        if line!=['']:
            if line[1]!=line[2]:
                # Add 'sender' into the graph-entry of the 'recipient'
                if line[2] not in graph:
                    graph[line[2]]={"pr":0,"rf":[line[1]],"st":[],"st_count":0, "rf_count":0, "hub":0, "auth":0}
                elif line[2] in graph:
                    graph[line[2]]["rf"].append(line[1])
                # Add 'recipients' into the graph-entry of 'sender'
                if line[1] not in graph:
                    graph[line[1]]={"pr":0,"rf":[],"st":[line[2]],"st_count":0, "rf_count":0, "hub":0, "auth":0}
                elif line[1] in graph:
                    graph[line[1]]["st"].append(line[2])
    for person in graph:
        graph[person]['st_count']=len(graph[person]['st'])
        graph[person]['rf_count']=len(graph[person]['rf'])
    nodes=len(graph)
    for person in graph:
        graph[person]['pr']=1./nodes
        graph[person]['auth']=1./math.sqrt(nodes)
        graph[person]['hub']=1./math.sqrt(nodes)
    return graph

# Iterate Pagerank algorithm
def iteratePagerank(graph,nodes):
    graph_fossil=copy.deepcopy(graph)
    sink_sum=0
    for address in graph:
        if graph[address]['st_count']==0:
            sink_sum+=graph[address]['pr']
    random=(1-LAMDA+(LAMDA*sink_sum))/nodes # Random hopping element
    for address in graph_fossil:
        # Calculate the PR sharing section (i.e. find 'share')
        sharesum=0
        for incoming in graph_fossil[address]['rf']: # Loop through all people sending PR to this address
            sharesum+=graph_fossil[incoming]['pr']/graph_fossil[incoming]['st_count']
        share=LAMDA*sharesum
        graph[address]['pr']=random+share
    return graph

# Iterate the HITS algorithm (Hubs first, then Authorities)
def iterateHITS(graph,nodes):    
    
    # HUBS...
    graph_fossil=copy.deepcopy(graph)
    for address in graph_fossil:
        hubsum=0
        # Calculate Hub score
        for outgoing in graph_fossil[address]['st']:
            hubsum+=graph_fossil[outgoing]['auth']
        graph[address]['hub']=hubsum
    total_hubs_sq=0
    # Find normaliser
    for address in graph:
        total_hubs_sq+=((graph[address]['hub'])**2)
    # Normalise
    for address in graph:
        graph[address]['hub']=graph[address]['hub']/math.sqrt(total_hubs_sq)
    
    # AUTHORITIES...
    graph_fossil=copy.deepcopy(graph)
    for address in graph_fossil:
        authsum=0
        # Calculate Authority score
        for incoming in graph_fossil[address]['rf']:
            authsum+=graph_fossil[incoming]['hub']
        graph[address]['auth']=authsum
    total_auth_sq=0
    # Find normaliser
    for address in graph:
        total_auth_sq+=((graph[address]['auth'])**2)
    # Normalise
    for address in graph:
        graph[address]['auth']=graph[address]['auth']/math.sqrt(total_auth_sq)

    return graph

# Write output Pagerank file
def writePROutput(graph):
    pr_dict={}
    for address in graph:
        pr_dict[address]=graph[address]['pr']
    output_pr = sorted(pr_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    f=open('pr.txt','w')
    for i in range(10):
        f.write('%.8f %s\n' % (output_pr[i][1],output_pr[i][0]))

# Write output 'Auth' and 'Hubs' files
def writeHITSOutput(graph):
    hub_dict={}
    auth_dict={}
    for address in graph:
        hub_dict[address]=graph[address]['hub']
        auth_dict[address]=graph[address]['auth']
    output_hub = sorted(hub_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    #f=open('full_hub.txt','w')
    #for i in range(len(output_hub)):
    #    f.write('%.8f %s\n' % (output_hub[i][1],output_hub[i][0]))
    f=open('hub.txt','w')
    for i in range(10):
        f.write('%.8f %s\n' % (output_hub[i][1],output_hub[i][0]))
    output_auth = sorted(auth_dict.iteritems(), key=operator.itemgetter(1), reverse=True)
    #f=open('full_auth.txt','w')
    #for i in range(len(output_auth)):
    #    f.write('%.8f %s\n' % (output_auth[i][1],output_auth[i][0]))
    f=open('auth.txt','w')
    for i in range(10):
        f.write('%.8f %s\n' % (output_auth[i][1],output_auth[i][0]))

# Saves final graph output to a database        
def saveGraph(graph,filename):
    tts4=shelve.open(filename)
    tts4["graph"]=graph

# =========== PAGERANK / HITS MAIN LOOP ============
print "Parsing data..."
parsed_data=parseData("graph.txt")
print "Drawing graph..."
graph=drawLinks(parsed_data)

if PR:
    nodes=len(graph)
    for i in range(ITERATIONS):
        print "Pagerank iteration number "+str(i+1)+"..."
        graph=iteratePagerank(graph,nodes)
    print "Writing output..."
    writePROutput(graph)
    print "Done.\n"  

if HITS:
    for i in range(ITERATIONS-1):
        print "HITS iteration number "+str(i+1)+"..."
        graph=iterateHITS(graph,nodes)
    if SAVE_DATABASE==True:
        print "Saving graph..."
        saveGraph(graph,'full_graph.db')
    print "Writing output..."
    writeHITSOutput(graph)
    print "Done.\n"
    
    
# =========== INITIAL GRAPH VARIABLE VALUES ============

SENIOR_ARROW_TH=14
ARROW_TH=39
BOLD_TH=100
RATIO1=0.6
RATIO2=1.0
RATIO3=1.4

# =========== GRAPH DRAWING FUNCTIONS ============

# Print viz_data dictionary 
def printViz_data(viz_data):       
    for email in viz_data:
        for key in viz_data[email]:
            print email,key,viz_data[email][key]
            
# Create data structure to just include people for visualisation ({EMAIL1:{st_r:xx, rf_r:xx, ratio:xx, EMAILX:(#ST,#RF), EMAILY:(#ST,#RF)}, EMAIL2:...)
def createVizData(emails,senior,graph):
    viz_data={}
    for email in emails:
        # Create main structure
        viz_data[email]={'st_r':0,'rf_r':0,'ratio_r':0,'name':email.replace(".","").replace("@","")}
        # Create 'full-counts' of emails sent and received (based on all records)
        viz_data[email]['st_f']=graph[email]['st_count']
        viz_data[email]['rf_f']=graph[email]['rf_count']
        if viz_data[email]['rf_f']==0:
            viz_data[email]['ratio_f']=1000
        else: viz_data[email]['ratio_f']=float(viz_data[email]['st_f'])/float(viz_data[email]['rf_f'])
        for sub_email in emails:
            viz_data[email][sub_email]=0
    # Add in count of times EMAIL sent mail to SUB_EMAIL and received mail from SUB_EMAIL (as a tuple of counts)
    for email in viz_data:
        for sub_email in emails:
            if sub_email!=email:
                viz_data[email][sub_email]=graph[email]["st"].count(sub_email)
    # Create 'reduced-counts' of emails sent and received (based on reduced set)
    for email in viz_data:
        for sub_email in emails:
            viz_data[email]['st_r']+=viz_data[email][sub_email]
            viz_data[email]['rf_r']+=viz_data[sub_email][email]
        if viz_data[email]['rf_r']==0:
            viz_data[email]['ratio_r']=1000
        else: viz_data[email]['ratio_r']=float(viz_data[email]['st_r'])/float(viz_data[email]['rf_r'])
    # Add seniority tag
    for email in viz_data:
        viz_data[email]['senior']=0
        if email in senior:
            viz_data[email]['senior']=1
    return viz_data

#Creates a dot file
def createDot(viz_data):
    f=open("graph.top",'w')
    f.write("digraph G {")
    # Create nodes
    for email in viz_data:
        sent_ratio=viz_data[email]["ratio_f"]
        if sent_ratio<RATIO1:
            colour="darkorange4"
        elif sent_ratio<RATIO2:
            colour="darkorange2"
        elif sent_ratio<RATIO3:
            colour="darkgoldenrod1"
        else: colour="yellow"   
        f.write("  "+viz_data[email]['name'])
        f.write(" [label = \""+email.replace("@enron.com","")+"\"")
        f.write(", style=\"filled\", color=\""+colour+"\"]")
        f.write("\n")
    f.write("\n")
    # Print arrows
    for email in viz_data:
        for sub_email in emails:
            if (viz_data[email]["senior"]==0 and viz_data[email][sub_email]>ARROW_TH) or (viz_data[email]["senior"]==1 and viz_data[email][sub_email]>SENIOR_ARROW_TH):
                f.write("  "+viz_data[email]['name']+" -> "+sub_email.replace(".","").replace("@",""))
                style="normal"
                number=viz_data[email][sub_email]
                if number>BOLD_TH:
                    style="bold"
                f.write(" [style="+style+", label=\""+str(number)+"\"];")
                f.write("\n")
    f.write("}")
    
# =========== MAIN LOOP FOR GRAPH DRAWING ============
            
auths=["ryan.slinger@enron.com","albert.meyers@enron.com","mark.guzman@enron.com","geir.solberg@enron.com","craig.dean@enron.com","bill.williams@enron.com","john.anderson@enron.com","michael.mier@enron.com","leaf.harasin@enron.com","eric.linder@enron.com"]
hubs=["pete.davis@enron.com", "bill.williams@enron.com", "rhonda.denton@enron.com","l..denton@enron.com","grace.rodriguez@enron.com","alan.comnes@enron.com","kathryn.sheppard@enron.com","kate.symes@enron.com","kysa.alport@enron.com","carla.hoffman@enron.com"]
pr=["klay@enron.com","jeff.skilling@enron.com","sara.shackleton@enron.com","tana.jones@enron.com","mark.taylor@enron.com","kenneth.lay@enron.com","louise.kitchen@enron.com","gerald.nemec@enron.com","jeff.dasovich@enron.com","sally.beck@enron.com"]
initial_handpicked=["david.delainey@enron.com","john.lavorato@enron.com","geoff.storey@enron.com","jonathan.mckay@enron.com","greg.whalley@enron.com","jeffrey.shankman@enron.com","mark.haedicke@enron.com","t..hodge@enron.com","mike.grigsby@enron.com","phillip.allen@enron.com","rick.buy@enron.com","vince.kaminski@enron.com","andy.zipper@enron.com","barry.tycholiz@enron.com","fletcher.sturm@enron.com","hunter.shively@enron.com","james.steffes@enron.com","john.arnold@enron.com","john.zufferli@enron.com","richard.shapiro@enron.com","scott.neal@enron.com","steven.kean@enron.com","elizabeth.sager@enron.com","stephanie.panus@enron.com"]
final_pr=["jeff.skilling@enron.com","sara.shackleton@enron.com","tana.jones@enron.com","mark.taylor@enron.com","kenneth.lay@enron.com","louise.kitchen@enron.com","jeff.dasovich@enron.com",]
final_handpicked=["david.delainey@enron.com","john.lavorato@enron.com","greg.whalley@enron.com","jeffrey.shankman@enron.com","mark.haedicke@enron.com","rick.buy@enron.com","vince.kaminski@enron.com","james.steffes@enron.com","john.arnold@enron.com","richard.shapiro@enron.com","steven.kean@enron.com","elizabeth.sager@enron.com"]
emails=set(final_pr+final_handpicked)
senior=["david.delainey@enron.com","jeff.skilling@enron.com","john.lavorato@enron.com","kenneth.lay@enron.com","w..delainey@enron.com","andrew.lewis@enron.com","f..brawner@enron.com","frank.ermis@enron.com","geoff.storey@enron.com","h..lewis@enron.com","jonathan.mckay@enron.com","keith.holst@enron.com","kevin.hyatt@enron.com","larry.may@enron.com","matt.motley@enron.com","mike.maggi@enron.com","robert.badeer@enron.com","robert.benson@enron.com","sandra.brawner@enron.com","vince.kaminski@enron.com"]

if VISUALIZE==True:
    print "Creating reduced set for vizualisation..."
    viz_data=createVizData(emails,senior,graph)
    print "Creating dot file..."
    createDot(viz_data)
    
print "Done, the program will now exit."
