function val2  = val2str9(x)
s = sprintf('%.10e',x);

    pos = strfind(s, 'e');

    ex = str2double(s(pos+2:end));

    tot = sprintf('%.9f', x*10^ex);
    val = (str2double(tot(strfind(tot,'.')+1:end)));
    val2=dec2bin(val);
    %valc=dec2bin(mod((val),2^4),4)
end
