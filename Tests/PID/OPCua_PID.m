function out = OPCua_PID(in, varargin)

    p = inputParser;
    addRequired(p, 'in')
    addParameter(p, 'init', false, @islogical)    
    addParameter(p, 'finish', false, @islogical)
    addParameter(p, 'update', false, @islogical)
    addParameter(p, 'reset', false, @islogical)
    addParameter(p, 'KP', 1., @isnumeric)
    addParameter(p, 'TI', 1., @isnumeric)
    addParameter(p, 'TD', 1., @isnumeric)
    parse(p, in, varargin{:}); 
    
    out = [0, 0, 0, 0, 0, 0];
   
    persistent OPCclient;  
    persistent nSignal nSet_point nKP nTI nTD nLL nLH nManual_mode nU_manual nReset ...
               nLimited nU
    % debug
    persistent nP nI nD nControl_signal
    
    if p.Results.init               
        % get client infos
        OPCclient = opcua('127.0.0.1', 4840);
        OPCclient.connect();
        % Init Vars
        % Get correct root node (since CoDeSys has something doubled)
        Root = findNodeByName(OPCclient.Namespace, 'DeviceSet', '-once');
        PrgNode = findNodeByName(Root, 'PIDtest', '-once');

        % get variables:
        nSignal = findNodeByName(PrgNode, 'signal', '-once');
        nSet_point = findNodeByName(PrgNode, 'set_point', '-once');
        nKP = findNodeByName(PrgNode, 'KP', '-once');
        nTI = findNodeByName(PrgNode, 'TI', '-once');
        nTD = findNodeByName(PrgNode, 'TD', '-once');
        nLL = findNodeByName(PrgNode, 'LL', '-once');
        nLH = findNodeByName(PrgNode, 'LH', '-once');
        nManual_mode = findNodeByName(PrgNode, 'manual_mode', '-once');
        nU_manual = findNodeByName(PrgNode, 'u_manual', '-once');
        nReset = findNodeByName(PrgNode, 'reset', '-once');
        nLimited = findNodeByName(PrgNode, 'limited', '-once');
        nU = findNodeByName(PrgNode, 'u', '-once');
        % debug
        nP = findNodeByName(PrgNode, 'P', '-once');
        nI = findNodeByName(PrgNode, 'I', '-once');
        nD = findNodeByName(PrgNode, 'D', '-once');
        nControl_signal = findNodeByName(PrgNode, 'control_signal', '-once');
    elseif p.Results.update
        nKP.writeValue(p.Results.KP);
        nTI.writeValue(p.Results.TI);
        nTD.writeValue(p.Results.TD);
        nLL.writeValue(-1.0);
        nLH.writeValue(1.0);
        % safely clear reset
        nReset.writeValue(0);
    elseif p.Results.reset
        % reset controller integrator
        nReset.writeValue(1);
        nSignal.writeValue(0);
        nSet_point.writeValue(0);
    elseif p.Results.finish
        % Reset Controller
        nSignal.writeValue(0);
        nSet_point.writeValue(0);
        nManual_mode.writeValue(0);
        nU_manual.writeValue(0);
        nReset.writeValue(1);
        % Close connection
        OPCclient.disconnect();
        % clean up
        OPCclient.delete();
        clear nSignal nSet_point nKP nTN nTV nLL nLH nManual_mode nU_manual nReset ...
              nLimited nU
        % debug
        clear nP nI nD nControl_signal
    else
        nSignal.writeValue(in(1));
        nSet_point.writeValue(in(2));
        nManual_mode.writeValue(in(3));
        nU_manual.writeValue(in(4));
        nReset.writeValue(in(5));
        out(1) = nU.readValue();
        out(2) = nLimited.readValue();
        % debug
        out(3) = nP.readValue();
        out(4) = nI.readValue();
        out(5) = nD.readValue();
        out(6) = nControl_signal.readValue();
    end
end

