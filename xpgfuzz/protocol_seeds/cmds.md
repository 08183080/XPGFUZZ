# smtp
python seed_enricher.py `
    --protocol-name "Exim SMTP" `
    --commands "HELO,EHLO,MAIL FROM,RCPT TO,DATA,QUIT,RSET,NOOP,VRFY,ETRN,BDAT,CHUNKING,DSN,PIPELINING,SIZE,STARTTLS" `
    --input-dir "in-smtp" `
    --output-dir "in-smtp-x-rag" `
    --variations 10


# proftp
python seed_enricher.py `
    --protocol-name "ProFTPD" `
    --commands "USER,PASS,ACCT,CWD,XCWD,CDUP,XCUP,SMNT,REIN,QUIT,PORT,EPRT,PASV,EPSV,TYPE,STRU,MODE,RETR,STOR,STOU,APPE,ALLO,REST,RNFR,RNTO,ABOR,DELE,MDTM,RMD,XRMD,MKD,MLSD,MLST,XMKD,PWD,XPWD,SIZE,LIST,NLST,SITE,SYST,STAT,HELP,NOOP,FEAT,OPTS,LANG,ADAT,AUTH,CCC,CONF,ENC,MIC,PBSZ,PROT,MFF,MFMT,HOST,CLNT,RANG,CSID" `
    --input-dir "in-proftpd" `
    --output-dir "in-proftpd-xx" `
    --variations 50


# pureftp
python seed_enricher.py `
    --protocol-name "PureFTPD" `
    --commands "ABOR,ALLO,APPE,AUTH TLS,CCC,CDUP,CWD,DELE,EPRT,EPSV,ESTA,ESTP,FEAT,HELP,LIST,MDTM,MFMT,MKD,MLSD,MLST,MODE,NLST,NOOP,PASS,PASV,PBSZ,PORT,PROT,PWD,QUIT,REST,RETR,RMD,RNFR,RNTO,SIZE,STAT,STOR,STOU,STRU,SYST,TYPE,USER,XCUP,XCWD,XDBG,XMKD,XPWD,XRMD,OPTS MLST,SITE CHMOD,SITE HELP,SITE IDLE,SITE TIME,SITE UTIME" `
    --input-dir "in-pureftpd" `
    --output-dir "in-pureftpd-xx" `
    --variations 50

# lightftp
python seed_enricher.py `
    --protocol-name "LightFTPD" `
    --commands "USER,QUIT,NOOP,PWD,TYPE,PORT,LIST,CDUP,CWD,RETR,ABOR,DELE,PASV,PASS,REST,SIZE,MKD,RMD,STOR,SYST,FEAT,APPE,RNFR,RNTO,OPTS,MLSD,AUTH,PBSZ,PROT,EPSV,HELP,SITE" `
    --input-dir "in-lightftpd" `
    --output-dir "in-lightftpd-xx" `
    --variations 100

# live555
python seed_enricher.py `
    --protocol-name "RTSP live555" `
    --commands "OPTIONS,DESCRIBE,SETUP,PLAY,PAUSE,TEARDOWN,GET_PARAMETER,SET_PARAMETER,REGISTER,DEREGISTER,ANNOUNCE,RECORD" `
    --input-dir "in-live555" `
    --output-dir "in-live555-xx" `
    --variations 20

# sip/kamilio
python seed_enricher.py `
    --protocol-name "sip kamailio" `
    --commands "INVITE,ACK,BYE,CANCEL,REGISTER,OPTIONS,INFO,PRACK,UPDATE,SUBSCRIBE,NOTIFY,PUBLISH,MESSAGE,REFER" `
    --input-dir "in-kamailio" `
    --output-dir "in-kamailio-xx" `
    --variations 100


# 提示词
问：deepwiki
给我返回live555支持的所有命令，用一行表示，命令间用,作为间隔。
