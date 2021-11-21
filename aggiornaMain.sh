#!/bin/bash

#Questo script aggiorna il main aggiungendo le modifiche fatte al branch

#Si sposta nella directory di git
cd ~/Documenti/GitHub_projects/ADM-HW4

#aggiorna il branch
./updateBranch.sh

#seleziona il branch da modificare
git checkout main

#esegue il merge dei due branch
git merge ame

#carica le modifiche fatte nello stage		
git add * 

#Controlla che non siano stati eliminati dei file dalla directory		 
log=`git commit -m "aggiornato" | awk '/eliminato:/{print $2}'`

#se li trova li elimina da git		
for i in $log
	do
		git rm $i; git commit -m "aggiornato"
	
done

#aggiorna repository remota
git push origin main
