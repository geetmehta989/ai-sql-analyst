import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import { Box, Button, CircularProgress, Container, IconButton, Paper, Stack, TextField, Typography } from '@mui/material'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import SendIcon from '@mui/icons-material/Send'
import { DataGrid, GridColDef } from '@mui/x-data-grid'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { BACKEND_URL } from './config'

type Message = { role: 'user' | 'assistant'; content: string }

const API_BASE = (BACKEND_URL || '').replace(/\/?$/, '')

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [uploading, setUploading] = useState(false)
  const [asking, setAsking] = useState(false)
  const [tableColumns, setTableColumns] = useState<string[]>([])
  const [tableRows, setTableRows] = useState<any[]>([])
  const [chartSpec, setChartSpec] = useState<{ type: string; xKey?: string; yKeys?: string[] } | null>(null)

  const fileInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    // Log backend URL for debugging
    // eslint-disable-next-line no-console
    console.log('Using backend:', import.meta.env.VITE_REACT_APP_BACKEND_URL)
  }, [])

  const gridColumns: GridColDef[] = useMemo(() => tableColumns.map((c) => ({ field: c, headerName: c, flex: 1 })), [tableColumns])

  const onUpload = useCallback(async (file: File) => {
    const form = new FormData()
    form.append('file', file)
    setUploading(true)
    try {
      const res = await axios.post(`${API_BASE}/upload`, form, { headers: { 'Content-Type': 'multipart/form-data' } })
      const content = res?.data?.tables?.length
        ? `Loaded tables: ${(res.data.tables || []).join(', ')}`
        : `Uploaded: ${res?.data?.filename || file.name}`
      setMessages((prev) => [...prev, { role: 'assistant', content }])
    } catch (e: any) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `Upload failed: ${e?.response?.data?.detail || e.message}` }])
    } finally {
      setUploading(false)
    }
  }, [])

  const onAsk = useCallback(async () => {
    if (!question.trim()) return
    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setAsking(true)
    try {
      const res = await axios.post(`${API_BASE}/ask`, { question, top_k: 200 })
      const { answer, data, chart, sql } = res.data
      setMessages((prev) => [...prev, { role: 'assistant', content: `${answer}\n\nSQL: ${sql}` }])
      setTableColumns(data.columns)
      setTableRows((data.rows || []).map((r: any[], i: number) => Object.fromEntries(data.columns.map((c: string, idx: number) => [c, r[idx]]))))
      setChartSpec(chart)
    } catch (e: any) {
      setMessages((prev) => [...prev, { role: 'assistant', content: `Error: ${e?.response?.data?.detail || e.message}` }])
    } finally {
      setAsking(false)
      setQuestion('')
    }
  }, [question])

  return (
    <Container maxWidth="md" sx={{ py: 3 }}>
      <Stack spacing={2}>
        <Typography variant="h4">AI SQL Analyst</Typography>
        <Paper variant="outlined" sx={{ p: 2, height: 400, overflow: 'auto' }}>
          <Stack spacing={1}>
            {messages.map((m, idx) => (
              <Box key={idx} sx={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%' }}>
                <Paper sx={{ p: 1.5, bgcolor: m.role === 'user' ? 'primary.main' : 'grey.100', color: m.role === 'user' ? 'primary.contrastText' : 'inherit' }}>
                  <Typography variant="body2" style={{ whiteSpace: 'pre-wrap' }}>{m.content}</Typography>
                </Paper>
              </Box>
            ))}
          </Stack>
        </Paper>
        <Stack direction="row" spacing={1}>
          <input ref={fileInputRef} type="file" accept=".xls,.xlsx,.xlsm,.xlsb" style={{ display: 'none' }} onChange={(e) => e.target.files && onUpload(e.target.files[0])} />
          <Button variant="outlined" startIcon={<UploadFileIcon />} onClick={() => fileInputRef.current?.click()} disabled={uploading}>
            {uploading ? 'Uploading...' : 'Upload Excel'}
          </Button>
        </Stack>
        <Stack direction="row" spacing={1}>
          <TextField fullWidth placeholder="Ask a question about your data..." value={question} onChange={(e) => setQuestion(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && onAsk()} />
          <IconButton color="primary" onClick={onAsk} disabled={asking}>
            {asking ? <CircularProgress size={24} /> : <SendIcon />}
          </IconButton>
        </Stack>
        {tableColumns.length > 0 && (
          <Paper variant="outlined" sx={{ height: 300 }}>
            <DataGrid rows={tableRows.map((r, i) => ({ id: i, ...r }))} columns={gridColumns} density="compact" pageSizeOptions={[5, 10, 25, 50]} initialState={{ pagination: { paginationModel: { pageSize: 10 } } }} />
          </Paper>
        )}
        {chartSpec && chartSpec.type === 'bar' && chartSpec.xKey && chartSpec.yKeys && chartSpec.yKeys.length > 0 && (
          <Paper variant="outlined" sx={{ p: 2, height: 320 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={tableRows}>
                <XAxis dataKey={chartSpec.xKey} />
                <YAxis />
                <Tooltip />
                <Legend />
                {chartSpec.yKeys.map((k, idx) => (
                  <Bar key={k} dataKey={k} fill={["#8884d8", "#82ca9d", "#ffc658"][idx % 3]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        )}
      </Stack>
    </Container>
  )
}

