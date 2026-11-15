from playwright.sync_api import sync_playwright
with sync_playwright() as q:
 b=q.chromium.launch(headless=True,args=['--no-sandbox'])
 p=b.new_context(viewport={'width':1280,'height':1000},accept_downloads=True).new_page()
 errors=[];p.on('pageerror',lambda e:errors.append(str(e)))
 p.goto('file:///tmp/fashionshift/index.html')
 p.wait_for_selector('.tabs button',timeout=60000)
 # 1 default tab is Classify a garment
 assert p.locator('.tabs button.active').inner_text().startswith('Classify')
 assert p.locator('#import.panel.active').count()==1
 # 2 single image classification
 p.locator('#imgFile').set_input_files('/tmp/fs_test.png')
 p.wait_for_selector('#imgCtl',state='visible',timeout=15000)
 p.locator('#invert').check()
 p.locator('#runImport').click()
 p.wait_for_function("document.querySelector('#importResult').textContent.length>10",timeout=15000)
 r=p.locator('#importResult').inner_text()
 assert 'Trouser' in r,r
 p.screenshot(path='/tmp/fs_classify.png',full_page=True)
 # 3 CSV batch import
 p.locator('#csvFile').set_input_files('/tmp/fs_test.csv')
 p.wait_for_function("document.querySelector('#csvResult').textContent.length>10",timeout=15000)
 assert 'Trouser' in p.locator('#csvResult').inner_text()
 # 4 bench
 p.locator('.tabs button[data-tab=bench]').click()
 p.wait_for_selector('#benchGrid .sample',timeout=15000)
 assert p.locator('#benchGrid .sample').count()==24
 p.screenshot(path='/tmp/fs_bench.png',full_page=True)
 # 5 evidence
 p.locator('.tabs button[data-tab=evidence]').click()
 assert p.locator('#metricsTable tbody tr').count()==3
 assert p.locator('#failGrid .sample').count()>10
 p.screenshot(path='/tmp/fs_evidence.png',full_page=True)
 # 6 stress
 p.locator('.tabs button[data-tab=stress]').click()
 p.wait_for_selector('#stressGrid .sample',timeout=15000)
 p.locator('input[name=pert][value=blur]').check()
 p.wait_for_timeout(800)
 assert 'accuracy' in p.locator('#stressKpis').inner_text().lower() or p.locator('#stressKpis .kpi').count()>=3
 p.screenshot(path='/tmp/fs_stress.png',full_page=True)
 # 7 model card
 p.locator('.tabs button[data-tab=card]').click()
 assert 'Honest limitations' in p.locator('#cardMd').inner_text()
 p.screenshot(path='/tmp/fs_card.png',full_page=True)
 # 8 mobile
 m=b.new_context(viewport={'width':390,'height':844},is_mobile=True).new_page()
 m.goto('file:///tmp/fashionshift/index.html');m.wait_for_selector('.tabs button',timeout=60000)
 for t in ['import','bench','evidence','stress','card']:
  m.locator('.tabs button[data-tab='+t+']').click();m.wait_for_timeout(400)
  assert m.evaluate('document.documentElement.scrollWidth<=391'),t
 m.screenshot(path='/tmp/fs_mobile.png',full_page=False)
 assert not errors,errors
 b.close()
print('FashionShift UI checks passed')
