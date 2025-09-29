# Vercel React TS Starter（修正版）

這個樣板為 Vercel 最小可用配置：
- Framework: Vite
- Build: `vite build`
- Output: `dist`
- Node engines: `>=18 <22`（避免 Node 22 在某些情況下的相容性問題）

## 在 Vercel 部署（零指令做法）
1. 直接把這個資料夾的所有檔案上傳到 GitHub 新的 repo
2. 到 Vercel -> Add New -> Project -> 選該 repo
3. 設定：Framework=Vite, Build=`vite build`, Output=`dist`
4. Deploy，完成後即可分享網址

## 本地端（可選）
```bash
npm install
npm run dev
```
