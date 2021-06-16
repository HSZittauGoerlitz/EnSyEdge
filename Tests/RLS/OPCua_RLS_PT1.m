function out = OPCua_RLS_PT1(in, varargin)

    p = inputParser;
    addRequired(p, 'in')
    addParameter(p, 'init', false, @islogical)    
    addParameter(p, 'finish', false, @islogical)
    addParameter(p, 'reset', false, @islogical)
    parse(p, in, varargin{:}); 
    
    out = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
   
    persistent OPCclient;  
    persistent enableNode getValuesNode uNode yNode ...
               b0Node a1Node;
           
    % for debuging
    persistent dydt_in dydt gamma_1 gamma_2 P_11 P_12 P_21 P_22
    
    if p.Results.init               
        % get client infos
        OPCclient = opcua('127.0.0.1', 4840);
        OPCclient.connect('mag', 'MAR');
        % Init Vars
        % Get correct root node (since CoDeSys has something doubled)
        Root = findNodeByName(OPCclient.Namespace, 'DeviceSet', '-once');
        PrgNode = findNodeByName(Root, 'Debug_PT1', '-once');

        % get variables:
        enableNode = findNodeByName(PrgNode, 'sim_enable', '-once');
        getValuesNode = findNodeByName(PrgNode, 'getValues', '-once');
        uNode = findNodeByName(PrgNode, 'u', '-once');
        yNode = findNodeByName(PrgNode, 'y', '-once');
        b0Node = findNodeByName(PrgNode, 'b0', '-once');
        a1Node = findNodeByName(PrgNode, 'a1', '-once');
        % debug
        dydt_in = findNodeByName(PrgNode, 'Signal', '-once');
        dydt = findNodeByName(PrgNode, 'dSignal_dt', '-once');
        gamma_1 = findNodeByName(PrgNode, 'gamma[1]', '-once');
        gamma_2 = findNodeByName(PrgNode, 'gamma[2]', '-once');
        P_11 = findNodeByName(PrgNode, 'P[1,1]', '-once');
        P_12 = findNodeByName(PrgNode, 'P[1,2]', '-once');
        P_21 = findNodeByName(PrgNode, 'P[2,1]', '-once');
        P_22 = findNodeByName(PrgNode, 'P[2,2]', '-once');
    elseif p.Results.finish
        OPCclient.disconnect();
        OPCclient.delete();
        clear enableNode getValuesNode uNode yNode ...
              b0Node a1Node ...
              dydt_in dydt gamma_1 gamma_2 P_11 P_12 P_21 P_22;
    elseif p.Results.reset
        gamma_1.writeValue(0.0);
        gamma_2.writeValue(0.0);
        P_11.writeValue(0.0);
        P_12.writeValue(0.0);
        P_21.writeValue(0.0);
        P_22.writeValue(0.0);
    else
        enableNode.writeValue(in(1));
        getValuesNode.writeValue(in(2));
        uNode.writeValue(in(3));
        yNode.writeValue(in(4));
        out(1) = b0Node.readValue();
        out(2) = a1Node.readValue();
        % debug
        out(3) = dydt_in.readValue();
        out(4) = dydt.readValue();
        out(5) = gamma_1.readValue();     
        out(6) = gamma_2.readValue();
        out(7) = P_11.readValue();
        out(8) = P_12.readValue();
        out(9) = P_21.readValue();
        out(10) = P_22.readValue();
    end
end

